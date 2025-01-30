use bevy::prelude::*;

use crate::game::GamePlugin;

pub mod game;

fn main() {
    App::new().add_plugins((DefaultPlugins, GamePlugin)).run();
}
