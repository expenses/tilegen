use image::*;
use direction::CardinalDirection as Direction;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use structopt::StructOpt;
use std::fmt;

#[derive(Clone, PartialEq, Eq, Hash, serde::Deserialize)]
struct TileLabel {
	label: String,
	#[serde(default)]
	rotation: Rotation,
}

impl fmt::Debug for TileLabel {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.rotation == Rotation::Normal {
			write!(f, "{:?}", self.label)
		} else {
			write!(f, "({:?}, {:?})", self.label, self.rotation)
		}
    }
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

	fn opposite(self) -> Self {
		match self {
			Self::Normal => Self::Opposite,
			Self::Plus90 => Self::Minus90,
			Self::Opposite => Self::Normal,
			Self::Minus90 => Self::Plus90,
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
	rotatable: Rotatable,
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

impl InputParams {
	fn allowed_neighbours<'a>(&'a self, label: &'a TileLabel) -> impl Iterator<Item=(Direction, TileLabel)> + 'a {
		// Get the allowed neighbours for a label
		self.tiles[&label.label.to_string()].allowed_neighbours.iter()
			// Flat map in the case of `NeighbourDirection::All`.
			.flat_map(move |(direction, neighbour_labels)| {
				direction.into_cardinal().iter().map(move |dir| (dir, neighbour_labels)) 
			})
			// Rotate the direction by the label direction.
			.map(move |(dir, neighbour_labels)| (label.rotation.rotate(*dir), neighbour_labels))
			// Flatmap to neighbour labels, rotating them if the can be rotated.
			.flat_map(move |(direction, neighbour_labels)| {
				neighbour_labels.clone().into_iter()
					.map(move |mut neighbour_label| {
						if let Rotatable::Yes { .. } = self.tiles[&neighbour_label.label.to_string()].rotatable {
							neighbour_label.rotation = neighbour_label.rotation.apply(label.rotation);
						}

						(direction, neighbour_label)
					})
			})
			.flat_map(move |(direction, neighbour_label)| {
				if let Rotatable::Yes { symmetry: Symmetry::I } = self.tiles[&neighbour_label.label.to_string()].rotatable {
					let mut opposite_neighbour_label = neighbour_label.clone();
					opposite_neighbour_label.rotation = opposite_neighbour_label.rotation.opposite();
					vec![(direction, neighbour_label), (direction, opposite_neighbour_label)]
				} else {
					vec![(direction, neighbour_label)]
				}
			})
	}
}

#[derive(serde::Deserialize, Debug, Clone)]
enum Rotatable {
	Yes { symmetry: Symmetry },
	No
}

impl Default for Rotatable {
	fn default() -> Self {
		Self::No
	}
}

#[derive(serde::Deserialize, Debug, Clone)]
enum Symmetry {
	None,
	I,
}

struct OutputSize {
	width: u16,
	height: u16,
}

impl std::str::FromStr for OutputSize {
	type Err = String;

	fn from_str(string: &str) -> Result<Self, Self::Err> {
		let index = string.find("x").ok_or_else(|| "Missing 'x' seperator")?;
		let width = string[..index].parse::<u16>().map_err(|err| err.to_string())?;
		let height = string[index+1..].parse::<u16>().map_err(|err| err.to_string())?;
		Ok(Self { width, height })
	}
}

#[derive(StructOpt)]
struct Opt {
	input_config: PathBuf,
	#[structopt(short, long, default_value = "50x50")]
	output_size: OutputSize,
	#[structopt(short, long, default_value = "1000")]
	iterations: usize,
	output_image: PathBuf,
}

fn main() -> Result<(), anyhow::Error> {
	env_logger::init();

	let opt = Opt::from_args();

	let input: InputParams = ron::de::from_reader(std::fs::File::open(opt.input_config)?)?;

	let tileset_image = image::open(&input.tileset_image_path)?;

	let mut tile_labels = Vec::new();

	for (label, tile) in input.tiles.iter() {
		if let Rotatable::Yes { .. } = tile.rotatable {
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
		for (direction, neighbour_tile) in input.allowed_neighbours(&tile) {
			let hashset = map.entry(tile.clone())
				.or_insert_with(HashMap::new)
				.entry(direction)
				.or_insert_with(HashSet::new);

			log::debug!("Adding {:?} to the {:?} of {:?}", neighbour_tile, direction, tile);
			if hashset.insert(neighbour_tile) {
				additions += 1;
			}
		}
	}

	for tile in &tile_labels {
		for (direction, neighbour_tile) in input.allowed_neighbours(&tile) {
			let added = map.entry(neighbour_tile.clone())
				.or_insert_with(HashMap::new)
				.entry(direction.opposite())
				.or_insert_with(HashSet::new)
				.insert(tile.clone());

			if added {
				log::debug!("Commutatively adding {:?} to the {:?} of {:?}", tile, direction.opposite(), neighbour_tile);
				commutative_additions += 1;
			}
		}
	}

	log::info!("Additions: {}, Commutive additions: {}", additions, commutative_additions);

	let pattern_table = wfc::PatternTable::from_vec(
		tile_labels
			.iter()
			.map(|label| (label, &input.tiles[&label.label.to_string()]))
			.map(|(label, tile)| wfc::PatternDescription {
				weight: std::num::NonZeroU32::new(tile.weight),
				allowed_neighbours: direction::CardinalDirectionTable::new_array([
					map[label][&Direction::North].iter().map(label_to_u32).collect(),
					map[label][&Direction::East].iter().map(label_to_u32).collect(),
					map[label][&Direction::South].iter().map(label_to_u32).collect(),
					map[label][&Direction::West].iter().map(label_to_u32).collect(),
				])
			})
			.collect()
	);

	let mut rng = rand::thread_rng();
	let global_stats = wfc::GlobalStats::new(pattern_table);

	let rb = wfc::RunOwn::new(
		wfc::Size::new_u16(opt.output_size.width, opt.output_size.height),
		&global_stats,
		&mut rng
	);

	#[cfg(feature = "parallel")]
	let retry = wfc::retry::ParNumTimes(opt.iterations);
	#[cfg(not(feature = "parallel"))]
	let retry = wfc::retry::NumTimes(opt.iterations);

	let wave = rb.collapse_retrying(retry, &mut rng)
		.map_err(|wfc::PropagateError::Contradiction(c)| {
			let west = c.west.map(|id| &tile_labels[id as usize]);
			let north = c.north.map(|id| &tile_labels[id as usize]);
			let south = c.south.map(|id| &tile_labels[id as usize]);
			let east = c.east.map(|id| &tile_labels[id as usize]);
			anyhow::anyhow!("Contradiction occurred.\nNorth: {:?}\nEast: {:?}\nSouth: {:?}\nWest: {:?}", north, east, south, west)
		})?;

	let image = wave_to_image(
		wave, tile_size, &tile_labels, &tileset_image,
		|label| input.tiles[label].coords,
	);

	image.save(opt.output_image)?;
	Ok(())
}

fn wave_to_image(
	wave: wfc::Wave,
	tile_size: u32,
	tile_labels: &[TileLabel],
	tileset_image: &DynamicImage,
	coords: impl Fn(&str) -> (u32, u32),
) -> DynamicImage {
	let size = wave.grid().size();
	let mut rgba_image = image::RgbaImage::new(size.width() * tile_size, size.height() * tile_size);
	wave.grid().enumerate().for_each(|(wfc::Coord { x, y }, cell)| {
		let ((coord_x, coord_y), rotation) = match cell.chosen_pattern_id() {
			Ok(id) => {
				let tile = &tile_labels[id as usize];
				(coords(&tile.label), tile.rotation)
			},
			Err(wfc::ChosenPatternIdError::MultipleCompatiblePatterns(ids)) => {
				let tile = &tile_labels[ids[0] as usize];
				log::debug!("Got multiple patterns at ({}, {}): {:?}", x, y, ids.into_iter().map(|id| &tile_labels[id as usize]).collect::<Vec<_>>());
				(coords(&tile.label), tile.rotation)
			},
			_ => unreachable!()
		};
		let image = tileset_image.view(coord_x * tile_size, coord_y * tile_size, tile_size, tile_size);

		let image = match rotation {
			Rotation::Normal => image.to_image(),
			Rotation::Plus90 => image::imageops::rotate90(&image),
			Rotation::Opposite => image::imageops::rotate180(&image),
			Rotation::Minus90 => image::imageops::rotate270(&image)
		};

		let image_x = x * tile_size as i32;
		let image_y = y * tile_size as i32;

		image::imageops::overlay(
			&mut rgba_image,
			&image,
			image_x as u32, image_y as u32,
		)
	});
	DynamicImage::ImageRgba8(rgba_image)
}
