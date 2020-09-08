use image::*;
use direction::CardinalDirection as Direction;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use structopt::StructOpt;
use std::fmt;
use std::num::{NonZeroU16, NonZeroUsize, NonZeroU32};

#[derive(Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize, PartialOrd, Ord)]
struct TileLabel {
	label: String,
	#[serde(default, skip_serializing_if = "is_default")]
	rotation: Rotation,
	#[serde(default, skip_serializing_if = "is_default")]
	subsection: (u32, u32),
}

fn is_default<T: Default + PartialEq>(value: &T) -> bool {
	value == &Default::default()
}

impl TileLabel {
	fn new(label: &str, rotation: Rotation, subsection: (u32, u32)) -> Self {
		Self {
			label: label.to_string(),
			rotation, subsection,
		}
	}
}

impl fmt::Debug for TileLabel {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match (self.rotation, self.subsection) {
			(Rotation::Normal, (0, 0)) => write!(f, "{:?}", self.label),
			(Rotation::Normal, _) => write!(f, "({:?}, {:?})", self.label, self.subsection),
			(_, (0, 0)) => write!(f, "({:?}, {:?})", self.label, self.rotation),
			_ => write!(f, "({:?}, {:?}, {:?})", self.label, self.rotation, self.subsection)
		}
	}
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize, Debug, PartialOrd, Ord)]
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
	allowed_neighbours: HashMap<NeighbourDirection, Vec<TileLabel>>,
	weight: u32,
	coords: (u32, u32),
	#[serde(default)]
	dimensions: Dimensions,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct Dimensions(u32, u32);

impl Default for Dimensions {
	fn default() -> Self {
		Self(1, 1)
	}
}

#[derive(serde::Deserialize, Debug)]
struct InputParams {
	tileset_image_path: PathBuf,
	tile_size: NonZeroU32,
	tiles: HashMap<String, Tile>,
}

impl InputParams {
	fn allowed_neighbours<'a>(&'a self, label: &'a TileLabel)
		-> impl Iterator<Item=(Direction, TileLabel)> + 'a
	{
		// Get the allowed neighbours for a label
		self.tiles[&label.label.to_string()].allowed_neighbours.iter()
			// Flat map in the case of `NeighbourDirection::All`.
			.flat_map(move |(direction, neighbour_labels)| {
				direction.into_cardinal().iter().map(move |dir| (dir, neighbour_labels.clone()))
			})
			// Rewrite the neighbours of multi-sized tiles.
			.map(move |(direction, neighbour_labels)| {
				let Dimensions(width, height) = self.tiles[&label.label.to_string()].dimensions;

				match (direction, label.subsection, label.subsection.0 == width - 1, label.subsection.1 == height - 1) {
					// No change
					(Direction::North, (_, 0), _, _) | (Direction::West, (0, _), _, _) |
					(Direction::East, _, true, _) | (Direction::South, _, _, true) => (direction, neighbour_labels),

					(Direction::North, (x, y), _, _) => (direction, vec![TileLabel::new(&label.label, Rotation::Normal, (x, y - 1))]),
					(Direction::South, (x, y), _, _) => (direction, vec![TileLabel::new(&label.label, Rotation::Normal, (x, y + 1))]),
					(Direction::West, (x, y), _, _) => (direction, vec![TileLabel::new(&label.label, Rotation::Normal, (x - 1, y))]),
					(Direction::East, (x, y), _, _) => (direction, vec![TileLabel::new(&label.label, Rotation::Normal, (x + 1, y))]),
				}
			})
			// Rotate the direction by the label direction.
			.map(move |(dir, neighbour_labels)| (label.rotation.rotate(*dir), neighbour_labels))
			// Flatmap to neighbour labels
			.flat_map(move |(direction, neighbour_labels)| {
				neighbour_labels.clone().into_iter()
					.map(move |neighbour_label| (direction, neighbour_label))
			})
			// Rotating neighbour labels if the can be rotated.
			.map(move |(direction, mut neighbour_label)| {
				match self.tiles[&neighbour_label.label.to_string()].rotatable {
					Rotatable::Yes { .. } => {
						neighbour_label.rotation =
							neighbour_label.rotation.apply(label.rotation);
					},
					_ => {}
				}

				(direction, neighbour_label)
			})
			// Apply symmetry.
			.flat_map(move |(direction, neighbour_label)| {
				match self.tiles[&neighbour_label.label.to_string()].rotatable {
					Rotatable::Yes { symmetry: Symmetry::I } => {
						let mut opposite_neighbour_label = neighbour_label.clone();
						opposite_neighbour_label.rotation = opposite_neighbour_label.rotation.opposite();
						vec![(direction, neighbour_label), (direction, opposite_neighbour_label)]
					},
					Rotatable::Yes { symmetry: Symmetry::None } | Rotatable::No => {
						vec![(direction, neighbour_label)]
					},
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

#[derive(StructOpt)]
enum Subcommand {
	/// Generate a tilemap and save it to a file.
	GenerateTilemap {
		#[structopt(help = "The path to save the generated tilemap to as a .ron.")]
		path: PathBuf,
	},
	/// Generate a tilemap, convert it into an image and save it to a file.
	GenerateImage {
		#[structopt(help = "The path to save the generated image to as a .png.")]
		path: PathBuf,
	}
}

struct MapSize {
	width: NonZeroU16,
	height: NonZeroU16,
}

impl std::str::FromStr for MapSize {
	type Err = String;

	fn from_str(string: &str) -> Result<Self, Self::Err> {
		let index = string.find("x").ok_or_else(|| "Missing 'x' seperator")?;
		let width = string[..index].parse::<NonZeroU16>().map_err(|err| err.to_string())?;
		let height = string[index+1..].parse::<NonZeroU16>().map_err(|err| err.to_string())?;
		Ok(Self { width, height })
	}
}

#[derive(StructOpt)]
struct Opt {
	#[structopt(help = "The input configuration file.")]
	input_config: PathBuf,
	#[structopt(subcommand)]
	subcommand: Subcommand,
	#[structopt(short, long, default_value = "50x50", help = "The size of the generated tilemap.")]
	map_size: MapSize,
	#[structopt(short, long, default_value = "1000", help = "The number of times to try and make a valid tilemap.")]
	attempts: NonZeroUsize,
}

fn main() -> Result<(), anyhow::Error> {
	env_logger::init();

	let opt = Opt::from_args();

	let file = std::fs::File::open(&opt.input_config)
		.map_err(|err| anyhow::anyhow!("Could not open '{}': {}", opt.input_config.display(), err))?;

	let input: InputParams = ron::de::from_reader(file)
		.map_err(|err| anyhow::anyhow!("Failed to parse '{}': {}", opt.input_config.display(), err))?;

	let tileset_image_path = opt.input_config.parent()
		// We already establised that the input_config path is a file, not a directory.
		.unwrap_or_else(|| unreachable!())
		.join(&input.tileset_image_path);

	let tileset_image = image::open(&tileset_image_path)
		.map_err(|err| anyhow::anyhow!("Could not open '{}': {}", tileset_image_path.display(), err))?;

	// Generate all possible tile labels.
	let mut tile_labels: Vec<TileLabel> = input.tiles.iter()
		.flat_map(|(label, tile)| {
			if let Rotatable::Yes { .. } = tile.rotatable {
				vec![
					(label, tile, Rotation::Normal),
					(label, tile, Rotation::Minus90),
					(label, tile, Rotation::Plus90),
					(label, tile, Rotation::Opposite),
				]
			} else {
				vec![(label, tile, Rotation::Normal)]
			}
		})
		.flat_map(|(label, tile, rotation)| {
			let Dimensions(width, height) = tile.dimensions;

			(0 .. width)
				.flat_map(move |x| (0 .. height).map(move |y| (x, y)))
				.map(move |(x, y)| (label, rotation, (x, y)))
		})
		.map(|(label, rotation, subsection)| TileLabel { label: label.into(), rotation, subsection})
		.collect();

	tile_labels.sort_unstable();

	let label_to_u32 =  |label: &TileLabel| -> u32 {
		tile_labels.iter().enumerate().find(|(_, lb)| lb == &label).map(|(i, _)| i)
			.unwrap_or_else(|| panic!("Could not find {:?}", label)) as u32
	};

	let mut map: HashMap<TileLabel, HashMap<Direction, HashSet<TileLabel>>> = HashMap::new();

	let mut explicit_rules = 0;
	let mut commutive_rules = 0;

	for tile in &tile_labels {
		for (direction, neighbour_tile) in input.allowed_neighbours(&tile) {
			let hashset = map.entry(tile.clone())
				.or_insert_with(HashMap::new)
				.entry(direction)
				.or_insert_with(HashSet::new);

			log::debug!("Adding rule: {:?} to the {:?} of {:?}", neighbour_tile, direction, tile);
			if hashset.insert(neighbour_tile) {
				explicit_rules += 1;
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
				log::debug!(
					"Commutatively adding rule: {:?} to the {:?} of {:?}",
					tile, direction.opposite(), neighbour_tile,
				);
				commutive_rules += 1;
			}
		}
	}

	log::info!(
		"Explicit generated rules: {}, Commutively generated rules: {}",
		explicit_rules, commutive_rules,
	);

	let pattern_table = wfc::PatternTable::from_vec(
		tile_labels
			.iter()
			.map(|label| (label, &input.tiles[&label.label.to_string()]))
			.map(|(label, tile)| wfc::PatternDescription {
				weight: NonZeroU32::new(tile.weight),
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
		wfc::Size::new_u16(opt.map_size.width.get(), opt.map_size.height.get()),
		&global_stats,
		&mut rng
	);

	#[cfg(feature = "parallel")]
	let retry = wfc::retry::ParNumTimes(opt.attempts.get());
	#[cfg(not(feature = "parallel"))]
	let retry = wfc::retry::NumTimes(opt.attempts.get());

	let wave = rb.collapse_retrying(retry, &mut rng)
		.map_err(|wfc::PropagateError::Contradiction(c)| {
			let west = c.west.map(|id| &tile_labels[id as usize]);
			let north = c.north.map(|id| &tile_labels[id as usize]);
			let south = c.south.map(|id| &tile_labels[id as usize]);
			let east = c.east.map(|id| &tile_labels[id as usize]);
			anyhow::anyhow!(
				"All {} attempts reached an situation where no tile could be placed in a coord.\
				\nYou could try turning up the number of attempts with `--attempts`,\
				\nTurning the output size down with `--output-size`,\
				\nOr adding additional rules to account to resolve this situation.\
				\nHere's what was in each position around the coord:\
				\nNorth: {:?}\nEast: {:?}\nSouth: {:?}\nWest: {:?}",
				opt.attempts, north, east, south, west,
			)
		})?;

	let grid = wave_to_grid(&wave, &tile_labels);

	match opt.subcommand {
		Subcommand::GenerateTilemap { path } => {
			let file = std::fs::File::create(path)?;
			ron::ser::to_writer(file, &grid)?;
		},
		Subcommand::GenerateImage { path } => {
			let image = grid_to_image(
				&grid, input.tile_size.get(), &tileset_image,
				|label| {
					let (x, y) = input.tiles[&label.label].coords;
					let (sub_x, sub_y) = label.subsection;
					(x + sub_x, y + sub_y)
				},
			);

			image.save_with_format(path, image::ImageFormat::Png)?;
		}
	}

	Ok(())
}

fn wave_to_grid(
	wave: &wfc::Wave,
	tile_labels: &[TileLabel],
) -> grid_2d::Grid<TileLabel> {
	grid_2d::Grid::new_grid_map_ref_with_coord(wave.grid(), |wfc::Coord { x, y }, cell| {
		match cell.chosen_pattern_id() {
			Ok(id) => tile_labels[id as usize].clone(),
			Err(wfc::ChosenPatternIdError::MultipleCompatiblePatterns(ids)) => {
				let tile = tile_labels[ids[0] as usize].clone();
				log::debug!(
					"Got multiple patterns at ({}, {}): {:?}",
					x, y, ids.into_iter().map(|id| &tile_labels[id as usize]).collect::<Vec<_>>(),
				);
				tile
			},
			_ => unreachable!()
		}
	})
}

fn grid_to_image(
	grid: &grid_2d::Grid<TileLabel>,
	tile_size: u32,
	tileset_image: &DynamicImage,
	coords: impl Fn(&TileLabel) -> (u32, u32),
) -> image::RgbaImage {
	let size = grid.size();
	let mut rgba_image = image::RgbaImage::new(size.width() * tile_size, size.height() * tile_size);
	grid.enumerate().for_each(|(wfc::Coord { x, y }, label)| {
		let (coord_x, coord_y) = coords(label);

		let image = tileset_image.view(
			coord_x * tile_size, coord_y * tile_size, tile_size, tile_size,
		);

		let image = match label.rotation {
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
	rgba_image
}
