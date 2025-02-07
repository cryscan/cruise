use bevy::prelude::*;
use bevy_async_ecs::AsyncEcsPlugin;
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use crate::game::GamePlugin;

pub mod blueprint;
pub mod game;
pub mod llm;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, AsyncEcsPlugin, WorldInspectorPlugin::new()))
        .add_plugins(GamePlugin)
        .run();
}
