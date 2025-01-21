use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Card {
    Rock,
    Paper,
    Scissors,
}

#[derive(Debug, Default, Clone, Component, Reflect)]
#[reflect(Component)]
pub struct Deck {
    pub rock: usize,
    pub paper: usize,
    pub scissors: usize,
}

fn main() {
    App::new().add_plugins(DefaultPlugins).run();
}
