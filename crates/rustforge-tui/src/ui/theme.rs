use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::border;
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType, Borders};

use crate::app::{AppState, Palette};

pub const ASCII_BORDER: border::Set = border::Set {
    top_left: "+",
    top_right: "+",
    bottom_left: "+",
    bottom_right: "+",
    vertical_left: "|",
    vertical_right: "|",
    horizontal_top: "-",
    horizontal_bottom: "-",
};

#[derive(Clone, Copy)]
pub struct Theme {
    pub text: Color,
    pub accent: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub muted: Color,
    pub border: Color,
    /// Raw per-episode reward series.
    pub series_raw: Color,
    /// Smoothed reward series.
    pub series_trend: Color,
    /// Loss series.
    pub series_loss: Color,
    /// Policy signal series (epsilon, entropy, ...).
    pub series_policy: Color,
    pub ascii: bool,
}

impl Theme {
    pub fn for_app(app: &AppState) -> Self {
        let ascii = app.ascii();
        if app.no_color() || app.palette() == Palette::Monochrome {
            return Self {
                text: Color::Reset,
                accent: Color::Reset,
                success: Color::Reset,
                warning: Color::Reset,
                error: Color::Reset,
                muted: Color::Reset,
                border: Color::Reset,
                series_raw: Color::Reset,
                series_trend: Color::Reset,
                series_loss: Color::Reset,
                series_policy: Color::Reset,
                ascii,
            };
        }

        match app.palette() {
            Palette::HighContrast => Self {
                text: Color::White,
                accent: Color::LightCyan,
                success: Color::LightGreen,
                warning: Color::LightYellow,
                error: Color::LightRed,
                muted: Color::Gray,
                border: Color::White,
                series_raw: Color::LightYellow,
                series_trend: Color::LightCyan,
                series_loss: Color::LightMagenta,
                series_policy: Color::LightGreen,
                ascii,
            },
            Palette::Default | Palette::Monochrome => Self {
                text: Color::Reset,
                accent: Color::Cyan,
                success: Color::Green,
                warning: Color::Yellow,
                error: Color::Red,
                muted: Color::Gray,
                border: Color::DarkGray,
                series_raw: Color::Yellow,
                series_trend: Color::Cyan,
                series_loss: Color::Magenta,
                series_policy: Color::Green,
                ascii,
            },
        }
    }

    pub fn title(self) -> Style {
        Style::default()
            .fg(self.accent)
            .add_modifier(Modifier::BOLD)
    }

    pub fn text_style(self) -> Style {
        Style::default().fg(self.text)
    }

    pub fn muted_style(self) -> Style {
        Style::default().fg(self.muted)
    }

    /// Inverted badge that stays legible without color support.
    pub fn badge(self, color: Color) -> Style {
        Style::default()
            .fg(color)
            .add_modifier(Modifier::REVERSED | Modifier::BOLD)
    }

    /// Placeholder for an unknown value.
    pub fn dash(self) -> &'static str {
        if self.ascii {
            "-"
        } else {
            "—"
        }
    }

    pub fn block<'a>(self, title: impl Into<Line<'a>>, ascii: bool) -> Block<'a> {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .title_style(self.title())
            .border_style(Style::default().fg(self.border));
        if ascii {
            block.border_set(ASCII_BORDER)
        } else {
            block.border_type(BorderType::Rounded)
        }
    }
}
