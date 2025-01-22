use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Card {
    Rock,
    Paper,
    Scissors,
}

#[derive(Debug, Default, Clone, Component, Reflect)]
#[reflect(Component)]
pub struct Inventory {
    pub star: u32,
    pub coin: u32,
    pub rock: u32,
    pub paper: u32,
    pub scissors: u32,
}

#[derive(Debug, Default, Clone, Reflect, Deref, DerefMut)]
pub struct Guaranty(pub Inventory);

#[derive(Debug, Error)]
pub enum GuarantyError {
    #[error("you cannot take out {1} star(s) because you only have {0} star(s)")]
    Star(u32, u32),
    #[error("you cannot take out {1} coin(s) because you only have {0} coin(s)")]
    Coin(u32, u32),
    #[error("you cannot take out {1} rock card(s) because you only have {0} rock card(s)")]
    Rock(u32, u32),
    #[error("you cannot take out {1} paper card(s) because you only have {0} paper card(s)")]
    Paper(u32, u32),
    #[error("you cannot take out {1} scissors card(s) because you only have {0} scissors card(s)")]
    Scissors(u32, u32),
}

impl Inventory {
    pub fn check(&self, guaranty: &Guaranty) -> Result<(), GuarantyError> {
        match (self, guaranty) {
            (x, y) if x.star < y.star => Err(GuarantyError::Star(x.star, y.star)),
            (x, y) if x.coin < y.coin => Err(GuarantyError::Coin(x.star, y.star)),
            (x, y) if x.rock < y.rock => Err(GuarantyError::Rock(x.star, y.star)),
            (x, y) if x.paper < y.paper => Err(GuarantyError::Paper(x.star, y.star)),
            (x, y) if x.scissors < y.scissors => Err(GuarantyError::Scissors(x.star, y.star)),
            _ => Ok(()),
        }
    }

    pub fn receive(&mut self, guaranty: Guaranty) {
        self.star += guaranty.star;
        self.coin += guaranty.coin;
        self.rock += guaranty.rock;
        self.paper += guaranty.paper;
        self.scissors += guaranty.scissors;
    }
}

#[derive(Debug, Default, Clone, Component, Reflect)]
#[reflect(Component)]
pub struct Table {
    pub players: Option<[(Entity, Guaranty); 2]>,
}

fn main() {
    App::new().add_plugins(DefaultPlugins).run();
}
