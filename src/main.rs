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
    out: PathBuf,
}

#[derive(Debug, Clone, Resource, Reflect)]
pub struct Settings {
    pub url: String,
    pub output: PathBuf,
}

fn main() {
    let Args { url, out } = Args::parse();

    App::new()
        .add_plugins((DefaultPlugins, AsyncEcsPlugin, WorldInspectorPlugin::new()))
        .add_plugins(GamePlugin)
        .register_type::<Settings>()
        .insert_resource(Settings { url, output: out })
        .run();
}
