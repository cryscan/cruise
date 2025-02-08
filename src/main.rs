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
}

#[derive(Debug, Clone, Deref, DerefMut, Resource, Reflect)]
pub struct ServerUrl(pub String);

fn main() {
    let args = Args::parse();

    App::new()
        .add_plugins((DefaultPlugins, AsyncEcsPlugin, WorldInspectorPlugin::new()))
        .add_plugins(GamePlugin)
        .insert_resource(ServerUrl(args.url))
        .run();
}
