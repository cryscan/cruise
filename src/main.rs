use std::path::PathBuf;

use bevy::prelude::*;
use bevy_async_ecs::AsyncEcsPlugin;
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use clap::Parser;

use crate::game::GamePlugin;

pub mod game;
pub mod llm;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "http://localhost:65530")]
    url: String,
    #[arg(long, short, default_value = "./output")]
    output: PathBuf,
    #[arg(long, default_value = "64")]
    num_players: usize,
    #[arg(long, default_value = "16")]
    max_rounds: usize,
}

#[derive(Debug, Clone, Resource, Reflect)]
#[reflect(Resource)]
pub struct Settings {
    /// Base URL for the LLM API.
    pub url: String,
    /// Output directory.
    pub output: PathBuf,
    /// Number of players in the game.
    pub num_players: usize,
    /// Maximum rounds a player can play.
    pub max_rounds: usize,
}

fn main() {
    let Args {
        url,
        output,
        num_players,
        max_rounds,
    } = Args::parse();

    let settings = Settings {
        url,
        output,
        num_players,
        max_rounds,
    };

    App::new()
        .add_plugins((DefaultPlugins, AsyncEcsPlugin, WorldInspectorPlugin::new()))
        .add_plugins(GamePlugin)
        .register_type::<Settings>()
        .insert_resource(settings)
        .run();
}
