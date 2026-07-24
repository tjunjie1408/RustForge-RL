use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::border;
use ratatui::widgets::{Block, Borders};

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
}

impl Theme {
    pub fn for_app(app: &AppState) -> Self {
        if app.no_color() || app.palette() == Palette::Monochrome {
            return Self {
                text: Color::Reset,
                accent: Color::Reset,
                success: Color::Reset,
                warning: Color::Reset,
                error: Color::Reset,
                muted: Color::Reset,
            };
        }

        match app.palette() {
            Palette::HighContrast => Self {
                text: Color::White,
                accent: Color::Cyan,
                success: Color::Green,
                warning: Color::Yellow,
                error: Color::Red,
                muted: Color::Gray,
            },
            Palette::Default | Palette::Monochrome => Self {
                text: Color::Gray,
                accent: Color::LightBlue,
                success: Color::LightGreen,
                warning: Color::LightYellow,
                error: Color::LightRed,
                muted: Color::DarkGray,
            },
        }
    }

    pub fn title(self) -> Style {
        Style::default()
            .fg(self.accent)
            .add_modifier(Modifier::BOLD)
    }

    pub fn block<'a>(self, title: &'a str, ascii: bool) -> Block<'a> {
        let mut block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .title_style(self.title())
            .border_style(Style::default().fg(self.muted));
        if ascii {
            block = block.border_set(ASCII_BORDER);
        }
        block
    }
}
