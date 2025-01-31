use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const NUM_TABLES: usize = 32;
pub const NUM_PLAYERS: usize = 64;

const NAMES: &str = include_str!("names.txt");

#[derive(Debug, Default)]
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Inventory>()
            .register_type::<Player>()
            .register_type::<Table>()
            .register_type::<Id>()
            .add_systems(Startup, setup_scene);
    }
}

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

#[derive(Debug, Default, Clone, Reflect, Serialize, Deserialize)]
#[reflect(Default)]
pub struct Stake {
    pub star: u32,
    pub coin: u32,
    pub rock: u32,
    pub paper: u32,
    pub scissors: u32,
}

impl std::ops::Add<Stake> for Stake {
    type Output = Self;

    fn add(self, rhs: Stake) -> Self::Output {
        Stake {
            star: self.star + rhs.star,
            coin: self.coin + rhs.coin,
            rock: self.rock + rhs.rock,
            paper: self.paper + rhs.paper,
            scissors: self.scissors + rhs.scissors,
        }
    }
}

#[derive(Debug, Error)]
pub enum StakeError {
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
    pub fn take(&self, stake: &Stake) -> Result<Self, StakeError> {
        match (self, stake) {
            (x, y) if x.star < y.star => Err(StakeError::Star(x.star, y.star)),
            (x, y) if x.coin < y.coin => Err(StakeError::Coin(x.star, y.star)),
            (x, y) if x.rock < y.rock => Err(StakeError::Rock(x.star, y.star)),
            (x, y) if x.paper < y.paper => Err(StakeError::Paper(x.star, y.star)),
            (x, y) if x.scissors < y.scissors => Err(StakeError::Scissors(x.star, y.star)),
            (x, y) => Ok(Self {
                star: x.star - y.star,
                coin: x.coin - y.coin,
                rock: x.rock - y.rock,
                paper: x.paper - y.paper,
                scissors: x.scissors - y.scissors,
            }),
        }
    }

    pub fn receive(&self, stake: Stake) -> Self {
        Self {
            star: self.star + stake.star,
            coin: self.coin + stake.coin,
            rock: self.rock + stake.rock,
            paper: self.paper + stake.paper,
            scissors: self.scissors + stake.scissors,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Component, Reflect)]
#[reflect(Component)]
pub struct Player;

#[derive(Debug, Default, Clone, Component, Reflect)]
#[reflect(Component)]
pub enum Table {
    #[default]
    Empty,
    Occupied([Entity; 2]),
    Ready([(Entity, Stake); 2]),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deref, DerefMut, Component, Reflect)]
#[reflect(Component)]
pub struct Id(pub usize);

fn setup_scene(mut commands: Commands) {
    commands.spawn_batch(
        (0..NUM_TABLES).map(|index| (Id(index), Name::new("Table"), Table::default())),
    );

    let names: Vec<_> = NAMES.split("\n").collect();
    commands.spawn_batch(
        (0..NUM_PLAYERS).map(move |index| (Id(index), Name::new(names[index]), Player)),
    );
}
