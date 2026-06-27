//! Axum HTTP server: embedded frontend + (Task 5) the WebSocket endpoint.
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::header;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use std::time::Duration;
use tokio::sync::broadcast::error::RecvError;

use crate::state::AppState;

const INDEX_HTML: &str = include_str!("../static/index.html");
const APP_JS: &str = include_str!("../static/app.js");
const STYLE_CSS: &str = include_str!("../static/style.css");
const CHART_JS: &str = include_str!("../static/chart.min.js");

/// Build the Axum router (static assets now; `/ws` added in Task 5).
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/static/app.js", get(app_js))
        .route("/static/style.css", get(style_css))
        .route("/static/chart.min.js", get(chart_js))
        .route("/ws", get(ws_handler))
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn app_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/javascript")], APP_JS)
}

async fn style_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css")], STYLE_CSS)
}

async fn chart_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/javascript")], CHART_JS)
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let (snapshot, mut rx) = state.snapshot_and_subscribe();

    if send_json(&mut socket, &serde_json::json!({ "type": "snapshot", "rows": snapshot }))
        .await
        .is_err()
    {
        return;
    }

    let mut heartbeat = tokio::time::interval(Duration::from_secs(20));
    loop {
        tokio::select! {
            received = rx.recv() => match received {
                Ok(row) => {
                    let msg = serde_json::json!({ "type": "append", "row": row });
                    if send_json(&mut socket, &msg).await.is_err() {
                        break;
                    }
                }
                Err(RecvError::Lagged(_)) => {
                    // Slow client: resync from the authoritative in-memory history.
                    let (snap, new_rx) = state.snapshot_and_subscribe();
                    rx = new_rx;
                    let msg = serde_json::json!({ "type": "snapshot", "rows": snap });
                    if send_json(&mut socket, &msg).await.is_err() {
                        break;
                    }
                }
                Err(RecvError::Closed) => break,
            },
            _ = heartbeat.tick() => {
                if socket.send(Message::Ping(Vec::new())).await.is_err() {
                    break; // peer gone
                }
            }
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                _ => {} // Pong / Text / Binary from client: ignored
            }
        }
    }
}

async fn send_json(socket: &mut WebSocket, value: &serde_json::Value) -> Result<(), axum::Error> {
    socket.send(Message::Text(value.to_string())).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // for `oneshot`

    #[tokio::test]
    async fn serves_index_and_assets() {
        let app = router(AppState::new(16));

        let resp = app
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .oneshot(Request::builder().uri("/static/app.js").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn ws_sends_snapshot_then_append() {
        use crate::metrics::MetricRow;
        use futures_util::StreamExt;
        use tokio_tungstenite::tungstenite::Message as TMsg;

        let state = AppState::new(64);
        state.push(MetricRow { episode: 0, reward: 1.0, avg_loss: Some(0.5), epsilon: 0.9, global_step: 10 });

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = router(state.clone());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });

        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
            .await
            .unwrap();

        // First message: snapshot containing the pre-existing row.
        let first = ws.next().await.unwrap().unwrap();
        let text = match first { TMsg::Text(t) => t, other => panic!("expected text, got {other:?}") };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["type"], "snapshot");
        assert_eq!(v["rows"].as_array().unwrap().len(), 1);

        // Push a new row; expect an append.
        state.push(MetricRow { episode: 1, reward: 2.0, avg_loss: None, epsilon: 0.8, global_step: 20 });
        let second = loop {
            let m = ws.next().await.unwrap().unwrap();
            if let TMsg::Text(t) = m { break t; }
        };
        let v2: serde_json::Value = serde_json::from_str(&second).unwrap();
        assert_eq!(v2["type"], "append");
        assert_eq!(v2["row"]["episode"], 1);
        assert_eq!(v2["row"]["avg_loss"], serde_json::Value::Null); // None -> null
    }
}
