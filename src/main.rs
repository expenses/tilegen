use image::*;
use direction::CardinalDirection as Direction;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Clone, PartialEq, Eq, Hash, serde::Deserialize, Debug)]
struct TileLabel {
	label: String,
	#[serde(default)]
	rotation: Rotation,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, serde::Deserialize, Debug)]
enum Rotation {
	Normal,
	Minus90,
	Plus90,
	Opposite,
}

impl Rotation {
	fn rotate(self, direction: Direction) -> Direction {
		match self {
			Self::Normal => direction,
			Self::Minus90 => direction.left90(),
			Self::Plus90 => direction.right90(),
			Self::Opposite => direction.opposite(),
		}
	}

	fn apply(self, other: Self) -> Self {
		match (self, other) {
			(Self::Normal, other) => other,
			(this, Self::Normal) => this,
			(Self::Minus90, Self::Plus90) => Self::Normal,
			(Self::Plus90, Self::Minus90) => Self::Normal,
			(Self::Opposite, Self::Opposite) => Self::Normal,

			(Self::Minus90, Self::Minus90) => Self::Opposite,
			(Self::Plus90, Self::Plus90) => Self::Opposite,

			(Self::Minus90, Self::Opposite) => Self::Plus90,
			(Self::Opposite, Self::Minus90) => Self::Plus90,

			(Self::Plus90, Self::Opposite) => Self::Minus90,
			(Self::Opposite, Self::Plus90) => Self::Minus90,
		}
	}
}

impl Default for Rotation {
	fn default() -> Self {
		Self::Normal
	}
}

#[derive(PartialEq, Eq, Hash, serde::Deserialize, Debug, Clone, Copy)]
enum NeighbourDirection {
	All,
	Left,
	Right,
	Up,
	Down
}

impl NeighbourDirection {
	fn into_cardinal(self) -> &'static [Direction] {
		match self {
			Self::All => &[Direction::North, Direction::East, Direction::South, Direction::West],
			Self::Left => &[Direction::West],
			Self::Right => &[Direction::East],
			Self::Up => &[Direction::North],
			Self::Down => &[Direction::South],
		}
	}
}

#[derive(serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
struct Tile {
	#[serde(default)]
	rotatable: bool,
	#[serde(default)]
	allowed_neighbours: HashMap<NeighbourDirection, HashSet<TileLabel>>,
	weight: u32,
	coords: (u32, u32),
}

#[derive(serde::Deserialize, Debug)]
struct InputParams {
	tileset_image_path: PathBuf,
	tile_size: u32,
	tiles: HashMap<String, Tile>,
}

#[derive(StructOpt)]
struct Opt {
	input_config: PathBuf,
	//output_size: (u32, u32),
	output_image: PathBuf,
}

fn main() {
	env_logger::init();

	let opt = Opt::from_args();

	let input: InputParams = ron::de::from_reader(std::fs::File::open(opt.input_config).unwrap()).unwrap();

	let tiles_image = image::open(&input.tileset_image_path).unwrap();

	let mut tile_labels = Vec::new();

	for (label, tile) in input.tiles.iter() {
		if tile.rotatable {
			tile_labels.extend_from_slice(&[
				TileLabel { label: label.into(), rotation: Rotation::Normal },
				TileLabel { label: label.into(), rotation: Rotation::Minus90 },
				TileLabel { label: label.into(), rotation: Rotation::Plus90 },
				TileLabel { label: label.into(), rotation: Rotation::Opposite },
			]);
		} else {
			tile_labels.push(TileLabel { label: label.into(), rotation: Rotation::Normal });
		}
	}

	tile_labels.sort_unstable_by_key(|label| label.label.clone());

	let label_to_u32 =  |label: &TileLabel| -> u32 {
		tile_labels.iter().enumerate().find(|(_, lb)| lb == &label).map(|(i, _)| i)
			.unwrap_or_else(|| panic!("Could not find {:?}", label)) as u32
	};

	let tile_size = input.tile_size;

	let mut map: HashMap<TileLabel, HashMap<Direction, HashSet<TileLabel>>> = HashMap::new();

	let mut additions = 0;
	let mut commutative_additions = 0;

	for tile in &tile_labels {
		for (direction, neighbour_tiles) in input.tiles.get(&tile.label.to_string()).unwrap().allowed_neighbours.iter() {
			for direction in direction.into_cardinal().iter().map(|dir| tile.rotation.rotate(*dir)) {
				let hashset = map.entry(tile.clone())
					.or_insert_with(HashMap::new)
					.entry(direction)
					.or_insert_with(HashSet::new);

				for mut neighbour_tile in neighbour_tiles.clone() {
					if input.tiles.get(&neighbour_tile.label.to_string()).unwrap().rotatable {
						neighbour_tile.rotation = neighbour_tile.rotation.apply(tile.rotation);
					}
					log::info!("Adding {:?} to the {:?} of {:?}", neighbour_tile, direction, tile);
					if hashset.insert(neighbour_tile) {
						additions += 1;
					}
				}
			}
		}
	}

	for tile in &tile_labels {
		for (direction, neighbour_tiles) in input.tiles.get(&tile.label.to_string()).unwrap().allowed_neighbours.iter() {
			for direction in direction.into_cardinal().iter().map(|dir| tile.rotation.rotate(*dir)) {
				for mut neighbour_tile in neighbour_tiles.clone() {
					if input.tiles.get(&neighbour_tile.label.to_string()).unwrap().rotatable {
						neighbour_tile.rotation = neighbour_tile.rotation.apply(tile.rotation);
					}

					let added = map.entry(neighbour_tile.clone())
						.or_insert_with(HashMap::new)
						.entry(direction.opposite())
						.or_insert_with(HashSet::new)
						.insert(tile.clone());

					if added {
						log::info!("Commutatively adding {:?} to the {:?} of {:?}", tile, direction.opposite(), neighbour_tile);
						commutative_additions += 1;
					}
				}
			}
		}
	}

	log::info!("Additions: {}, Commutive additions: {}", additions, commutative_additions);

	let retry = wfc::retry::ParNumTimes(10_000);

	let pattern_table = wfc::PatternTable::from_vec(
		tile_labels
			.iter()
			.map(|label| (label, input.tiles.get(&label.label.to_string()).unwrap()))
			.map(|(label, tile)| wfc::PatternDescription {
				weight: std::num::NonZeroU32::new(tile.weight),
				allowed_neighbours: direction::CardinalDirectionTable::new_array([
					map.get(label).expect("north 1").get(&Direction::North).expect("north 2").iter().map(label_to_u32).collect(),
					map.get(label).expect("east 1").get(&Direction::East).expect("east 2").iter().map(label_to_u32).collect(),
					map.get(label).expect("south 1").get(&Direction::South).expect("south 2").iter().map(label_to_u32).collect(),
					map.get(label).expect("west 1").get(&Direction::West).expect("west 2").iter().map(label_to_u32).collect(),
				])
			})
			.collect()
	);

	for (key, value) in map.iter() {
		log::debug!("{:?}:\n\t{:?}", key, value);
	}

	let mut rng = rand::thread_rng();
	let global_stats = wfc::GlobalStats::new(pattern_table);

	let rb = wfc::RunOwn::new(
		wfc::Size::new_u16(100, 100),
		&global_stats,
		&mut rng
	);

	let wave = rb.collapse_retrying(retry, &mut rng).unwrap();

	let image = {
		let size = wave.grid().size();
		let mut rgba_image = image::RgbaImage::new(size.width() * tile_size, size.height() * tile_size);
		wave.grid().enumerate().for_each(|(wfc::Coord { mut x, mut y }, cell)| {
			x *= tile_size as i32;
			y *= tile_size as i32;

			let tile = match cell.chosen_pattern_id() {
				Ok(id) => &tile_labels[id as usize],
				_ => panic!(),
			};
			let image = input.tiles.get(&tile.label).unwrap();
			let (coord_x, coord_y) = image.coords;
			let image = tiles_image.view(coord_x * tile_size, coord_y * tile_size, tile_size, tile_size);

			let image = match tile.rotation {
				Rotation::Normal => image.to_image(),
				Rotation::Plus90 => image::imageops::rotate90(&image),
				Rotation::Opposite => image::imageops::rotate180(&image),
				Rotation::Minus90 => image::imageops::rotate270(&image)
			};

			image::imageops::overlay(
				&mut rgba_image,
				&image,
				x as u32, y as u32,
			)
		});
		DynamicImage::ImageRgba8(rgba_image)
	};

	image.save(opt.output_image).unwrap();
}
