use bevy::prelude::*;
use bevy_inspector_egui::quick::WorldInspectorPlugin;

use crate::game::GamePlugin;

pub mod game;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, WorldInspectorPlugin::new()))
        .add_plugins(GamePlugin)
        .run();
}
