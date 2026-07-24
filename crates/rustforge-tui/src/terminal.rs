//! Terminal ownership, preflight, and best-effort restoration.

use std::fmt;
use std::io::{self, IsTerminal, Stdout};
use std::panic;
use std::sync::Once;

use crossterm::cursor::{Hide, Show};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

pub const MIN_TERMINAL_WIDTH: u16 = 60;
pub const MIN_TERMINAL_HEIGHT: u16 = 18;

pub type DashboardTerminal = Terminal<CrosstermBackend<Stdout>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalPreflightError {
    InputNotTerminal,
    OutputNotTerminal,
}

impl fmt::Display for TerminalPreflightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputNotTerminal => formatter.write_str("standard input is not a terminal"),
            Self::OutputNotTerminal => formatter.write_str("standard output is not a terminal"),
        }
    }
}

impl std::error::Error for TerminalPreflightError {}

pub fn validate_terminal_environment(
    input_is_terminal: bool,
    output_is_terminal: bool,
) -> Result<(), TerminalPreflightError> {
    if !input_is_terminal {
        return Err(TerminalPreflightError::InputNotTerminal);
    }
    if !output_is_terminal {
        return Err(TerminalPreflightError::OutputNotTerminal);
    }
    Ok(())
}

pub fn preflight_current_terminal() -> Result<(), TerminalPreflightError> {
    validate_terminal_environment(io::stdin().is_terminal(), io::stdout().is_terminal())
}

pub fn validate_terminal_size(width: u16, height: u16) -> io::Result<()> {
    if width < MIN_TERMINAL_WIDTH || height < MIN_TERMINAL_HEIGHT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "terminal is {width}x{height}; at least {MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT} is required"
            ),
        ));
    }
    Ok(())
}

pub fn preflight_current_terminal_size() -> io::Result<()> {
    let (width, height) = crossterm::terminal::size()?;
    validate_terminal_size(width, height)
}

/// Owns raw mode and the alternate screen until explicitly restored or dropped.
pub struct TerminalGuard {
    terminal: DashboardTerminal,
    restored: bool,
}

impl TerminalGuard {
    pub fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnterAlternateScreen, Hide) {
            let _ = disable_raw_mode();
            return Err(error);
        }

        let backend = CrosstermBackend::new(stdout);
        let terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(error) => {
                restore_process_terminal();
                return Err(error);
            }
        };
        Ok(Self {
            terminal,
            restored: false,
        })
    }

    pub fn terminal_mut(&mut self) -> &mut DashboardTerminal {
        &mut self.terminal
    }

    pub fn restore(&mut self) -> io::Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;
        let raw_result = disable_raw_mode();
        let screen_result = execute!(self.terminal.backend_mut(), LeaveAlternateScreen, Show);
        let cursor_result = self.terminal.show_cursor();
        raw_result.and(screen_result).and(cursor_result)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// Best-effort restoration usable from the process panic hook.
pub fn restore_process_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen, Show);
}

/// Install the process-global restoration hook once, preserving the old hook.
///
/// Reusable library imports never call this implicitly; the executable runner
/// opts in immediately before entering the terminal.
pub fn install_terminal_panic_hook() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            restore_process_terminal();
            previous(info);
        }));
    });
}
