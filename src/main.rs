use image::*;
use direction::CardinalDirection as Direction;
use structopt::StructOpt;
use grid_2d::*;
use itertools::Itertools;
use rand::SeedableRng;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::fmt;
use std::num::{NonZeroU16, NonZeroUsize, NonZeroU32};

#[derive(Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize, PartialOrd, Ord)]
struct TileReference {
	label: String,
	#[serde(default, skip_serializing_if = "is_default")]
	rotation: Rotation,
	#[serde(default, skip_serializing_if = "is_default")]
	subsection: (u32, u32),
}

fn is_default<T: Default + PartialEq>(value: &T) -> bool {
	value == &Default::default()
}

impl TileReference {
	fn new(label: &str, rotation: Rotation, subsection: (u32, u32)) -> Self {
		Self {
			label: label.to_string(),
			rotation, subsection,
		}
	}
}

impl fmt::Debug for TileReference {
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
	allowed_neighbours: HashMap<NeighbourDirection, Vec<TileReference>>,
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
	fn allowed_neighbours<'a>(&'a self, reference: &'a TileReference)
		-> impl Iterator<Item=(Direction, TileReference)> + 'a
	{
		// Get the allowed neighbours for a reference
		self.tiles[&reference.label.to_string()].allowed_neighbours.iter()
			// Flat map in the case of `NeighbourDirection::All`.
			.flat_map(move |(direction, neighbour_refs)| {
				direction.into_cardinal().iter().map(move |dir| (dir, neighbour_refs.clone()))
			})
			// Rewrite the neighbours of multi-sized tiles.
			.map(move |(direction, neighbour_refs)| {
				let Dimensions(width, height) = self.tiles[&reference.label.to_string()].dimensions;

				match (direction, reference.subsection, reference.subsection.0 == width - 1, reference.subsection.1 == height - 1) {
					// No change
					(Direction::North, (_, 0), _, _) | (Direction::West, (0, _), _, _) |
					(Direction::East, _, true, _) | (Direction::South, _, _, true) => (direction, neighbour_refs),

					(Direction::North, (x, y), _, _) => (direction, vec![TileReference::new(&reference.label, Rotation::Normal, (x, y - 1))]),
					(Direction::South, (x, y), _, _) => (direction, vec![TileReference::new(&reference.label, Rotation::Normal, (x, y + 1))]),
					(Direction::West, (x, y), _, _) => (direction, vec![TileReference::new(&reference.label, Rotation::Normal, (x - 1, y))]),
					(Direction::East, (x, y), _, _) => (direction, vec![TileReference::new(&reference.label, Rotation::Normal, (x + 1, y))]),
				}
			})
			// Rotate the direction by the reference direction.
			.map(move |(dir, neighbour_refs)| (reference.rotation.rotate(*dir), neighbour_refs))
			// Flatmap to neighbour reference
			.flat_map(move |(direction, neighbour_refs)| {
				neighbour_refs.clone().into_iter()
					.map(move |neighbour_ref| (direction, neighbour_ref))
			})
			// Rotating neighbour references if the can be rotated.
			.map(move |(direction, mut neighbour_ref)| {
				match self.tiles[&neighbour_ref.label.to_string()].rotatable {
					Rotatable::Yes { .. } => {
						neighbour_ref.rotation =
							neighbour_ref.rotation.apply(reference.rotation);
					},
					_ => {}
				}

				(direction, neighbour_ref)
			})
			// Apply symmetry.
			.flat_map(move |(direction, neighbour_ref)| {
				match self.tiles[&neighbour_ref.label.to_string()].rotatable {
					Rotatable::Yes { symmetry: Symmetry::I } => {
						let mut opposite_neighbour_ref = neighbour_ref.clone();
						opposite_neighbour_ref.rotation = opposite_neighbour_ref.rotation.opposite();
						vec![(direction, neighbour_ref), (direction, opposite_neighbour_ref)]
					},
					Rotatable::Yes { symmetry: Symmetry::None } | Rotatable::No => {
						vec![(direction, neighbour_ref)]
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
		#[structopt(flatten)]
		generation_options: GenerationOptions,
		#[structopt(help = "The path to save the generated tilemap to as a .ron.")]
		path: PathBuf,
	},
	/// Generate a tilemap, convert it into an image and save it to a file.
	GenerateImage {
		#[structopt(flatten)]
		generation_options: GenerationOptions,
		#[structopt(help = "The path to save the generated image to as a .png.")]
		path: PathBuf,
	},
	/// Render a generated tilemap to an image.
	TilemapToImage {
		#[structopt(help = "The path of the tilemap.")]
		tilemap_path: PathBuf,
		#[structopt(help = "The path of the rendered image.")]
		image_path: PathBuf,
	}
}

fn parse_map_size(string: &str) -> Result<MapSize, anyhow::Error> {
	let index = string.find("x").ok_or_else(|| anyhow::anyhow!("Missing 'x' seperator"))?;
	let width = string[..index].parse::<NonZeroU16>()?;
	let height = string[index+1..].parse::<NonZeroU16>()?;
	Ok(MapSize { width, height })
}

struct MapSize {
	width: NonZeroU16,
	height: NonZeroU16,
}

#[derive(StructOpt)]
struct GenerationOptions {
	#[structopt(short, long, default_value = "50x50", parse(try_from_str = parse_map_size), help = "The size of the generated tilemap.")]
	map_size: MapSize,
	#[structopt(short, long, default_value = "1000", help = "The number of times to try and make a valid tilemap.")]
	attempts: NonZeroUsize,
	#[structopt(short, long, help = "The path to save an image to help in the event of a contradiction. By default an image will not be saved.")]
	contradiction_image_path: Option<PathBuf>,
	#[structopt(short, long, default_value = "None", parse(try_from_str = ron::de::from_str), help = "Which directions the map should wrap around in.")]
	wrap: Wrap,
	#[structopt(short, long, help = "A value to seed the random number generator with. By default it uses a seed generated by the operating system.")]
	seed: Option<u64>,
}

#[derive(StructOpt)]
struct Opt {
	#[structopt(help = "The input configuration file.")]
	input_config: PathBuf,
	#[structopt(subcommand)]
	subcommand: Subcommand,
}

#[derive(serde::Deserialize, Debug)]
enum Wrap {
	None, Vertically, Horizontally, Both,
}

fn main() -> Result<(), anyhow::Error> {
	env_logger::init();

	let opt = Opt::from_args();
	run(opt)
}

fn run(opt: Opt) -> Result<(), anyhow::Error> {
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

	match opt.subcommand {
		Subcommand::GenerateTilemap { path, generation_options } => {
			let grid = generate(&input, &tileset_image, &generation_options)?;
			let file = std::fs::File::create(path)?;
			ron::ser::to_writer(file, &grid)?;
		},
		Subcommand::GenerateImage { path, generation_options } => {
			let grid = generate(&input, &tileset_image, &generation_options)?;
			let image = grid_to_image(&grid, &input, &tileset_image);
			image.save_with_format(path, image::ImageFormat::Png)?;
		},
		Subcommand::TilemapToImage { tilemap_path, image_path } => {
			let tilemap_file = std::fs::File::open(tilemap_path)?;
			let tilemap = ron::de::from_reader(tilemap_file)?;
			let image = grid_to_image(&tilemap, &input, &tileset_image);
			image.save_with_format(image_path, image::ImageFormat::Png)?;
		}
	}

	Ok(())
}

// Generate all possible tile references.
fn generate_all_tile_references(input: &InputParams) -> Vec<TileReference> {
	let tile_refs: Vec<_> = input.tiles.iter()
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
		.map(|(label, rotation, subsection)| TileReference { label: label.into(), rotation, subsection})
		.sorted()
		.collect();

	log::info!("Expanded {} tiles in the input into {} tile references", input.tiles.len(), tile_refs.len());

	tile_refs
}

fn create_neighbour_map(input: &InputParams, tile_refs: &[TileReference]) -> HashMap<TileReference, HashMap<Direction, HashSet<TileReference>>> {
	let mut map: HashMap<TileReference, HashMap<Direction, HashSet<TileReference>>> = HashMap::new();

	let mut explicit_rules = 0;
	let mut commutive_rules = 0;

	for tile in tile_refs {
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

	for tile in tile_refs {
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

	map
}

fn generate(input: &InputParams, tileset_image: &DynamicImage, gen: &GenerationOptions) -> Result<Grid<TileReference>, anyhow::Error> {
	let tile_refs = generate_all_tile_references(input);

	let ref_to_u32 =  |reference: &TileReference| -> u32 {
		tile_refs.iter().enumerate().find(|(_, r)| r == &reference).map(|(i, _)| i)
			.unwrap_or_else(|| panic!("Could not find {:?}", reference)) as u32
	};

	let map = create_neighbour_map(input, &tile_refs);

	let get_ids_for_direction = |reference: &TileReference, direction: Direction| {
		map.get(reference)
			.unwrap_or_else(|| panic!("Can't find reference {:?}", reference))
			.get(&direction)
			.unwrap_or_else(|| panic!("{:?} has no neighbours in direction {:?}", reference, direction))
			.iter().sorted().map(ref_to_u32).collect()
	};

	let pattern_table = wfc::PatternTable::from_vec(
		tile_refs
			.iter()
			.sorted()
			.map(|reference| (reference, &input.tiles[&reference.label.to_string()]))
			.map(|(reference, tile)| wfc::PatternDescription {
				weight: NonZeroU32::new(tile.weight),
				allowed_neighbours: direction::CardinalDirectionTable::new_array([
					get_ids_for_direction(reference, Direction::North),
					get_ids_for_direction(reference, Direction::East),
					get_ids_for_direction(reference, Direction::South),
					get_ids_for_direction(reference, Direction::West),
				])
			})
			.collect()
	);

	let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(
		gen.seed.unwrap_or_else(rand::random)
	);
	let global_stats = wfc::GlobalStats::new(pattern_table);

	let wrap = match gen.wrap {
		Wrap::None => wfc::wrap::MultiWrap::None,
		Wrap::Vertically => wfc::wrap::MultiWrap::Y,
		Wrap::Horizontally => wfc::wrap::MultiWrap::X,
		Wrap::Both => wfc::wrap::MultiWrap::XY,
	};

	let rb = wfc::RunOwn::new_wrap(
		wfc::Size::new_u16(gen.map_size.width.get(), gen.map_size.height.get()),
		&global_stats,
		wrap,
		&mut rng
	);

	#[cfg(feature = "parallel")]
	let retry = wfc::retry::ParNumTimes(gen.attempts.get());
	#[cfg(not(feature = "parallel"))]
	let retry = wfc::retry::NumTimes(gen.attempts.get());

	let wave = match rb.collapse_retrying(retry, &mut rng) {
		Ok(wave) => wave,
		Err(wfc::PropagateError::Contradiction(c)) => {
			let possible_patterns = |contradiction_tile| {
				match contradiction_tile {
					wfc::ContradictionTile::ChosenPattern(id) => vec![&tile_refs[id as usize]],
					wfc::ContradictionTile::OffGrid | wfc::ContradictionTile::NoCompatiblePatterns => vec![],
					wfc::ContradictionTile::MultipleCompatiblePatterns(ids) => ids.iter().map(|&id| &tile_refs[id as usize]).collect()
				}
			};

			let west = possible_patterns(c.west);
			let north = possible_patterns(c.north);
			let south = possible_patterns(c.south);
			let east = possible_patterns(c.east);

			if let Some(path) = gen.contradiction_image_path.as_ref() {
				let tile_size = input.tile_size.get();
				let size = 3 * tile_size;
				let mut image = image::RgbaImage::new(size, size);

				image::imageops::overlay(
					&mut image,
					&merge_tiles(&input, &north, &tileset_image),
					tile_size, 0,
				);

				image::imageops::overlay(
					&mut image,
					&merge_tiles(&input, &west, &tileset_image),
					0, tile_size,
				);

				image::imageops::overlay(
					&mut image,
					&merge_tiles(&input, &east, &tileset_image),
					tile_size * 2, tile_size,
				);

				image::imageops::overlay(
					&mut image,
					&merge_tiles(&input, &south, &tileset_image),
					tile_size, tile_size * 2,
				);

				image.save_with_format(path, image::ImageFormat::Png)?;
			}

			return Err(anyhow::anyhow!(
				"A contradiction has been reached. This means that all {} attempts reached an\
				\nsituation where no tile could be placed in a coord.\
				\nYou could try turning up the number of attempts with `--attempts`,\
				\nTurning the map size down with `--map-size`,\
				\nOr adding additional rules to account to resolve this situation.\
				\nHere's what was in each position around the coord:\
				\nNorth: {:?}\nEast: {:?}\nSouth: {:?}\nWest: {:?}",
				gen.attempts, north, east, south, west,
			));
		}
	};

	Ok(wave_to_grid(&wave, &tile_refs))
}

fn wave_to_grid(
	wave: &wfc::Wave,
	tile_refs: &[TileReference],
) -> Grid<TileReference> {
	Grid::new_grid_map_ref_with_coord(wave.grid(), |wfc::Coord { x, y }, cell| {
		match cell.chosen_pattern_id() {
			Ok(id) => tile_refs[id as usize].clone(),
			Err(wfc::ChosenPatternIdError::MultipleCompatiblePatterns(ids)) => {
				let tile = tile_refs[ids[0] as usize].clone();
				log::debug!(
					"Got multiple patterns at ({}, {}): {:?}",
					x, y, ids.into_iter().map(|id| &tile_refs[id as usize]).collect::<Vec<_>>(),
				);
				tile
			},
			_ => unreachable!()
		}
	})
}

fn grid_to_image(
	grid: &Grid<TileReference>,
	input: &InputParams,
	tileset_image: &DynamicImage,
) -> image::RgbaImage {
	let size = grid.size();
	let tile_size = input.tile_size.get();
	let mut rgba_image = image::RgbaImage::new(size.width() * tile_size, size.height() * tile_size);
	grid.enumerate().for_each(|(wfc::Coord { x, y }, reference)| {
		let image = tile_to_image(input, reference, tileset_image);

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

fn tile_to_image(input: &InputParams, reference: &TileReference, tileset_image: &DynamicImage) -> RgbaImage {
	let (coord_x, coord_y) = {
		let (x, y) = input.tiles[&reference.label].coords;
		let (sub_x, sub_y) = reference.subsection;
		(x + sub_x, y + sub_y)
	};
	let tile_size = input.tile_size.get();

	let image = tileset_image.view(
		coord_x * tile_size, coord_y * tile_size, tile_size, tile_size,
	);

	match reference.rotation {
		Rotation::Normal => image.to_image(),
		Rotation::Plus90 => image::imageops::rotate90(&image),
		Rotation::Opposite => image::imageops::rotate180(&image),
		Rotation::Minus90 => image::imageops::rotate270(&image)
	}
}

fn merge_tiles(input: &InputParams, references: &[&TileReference], tileset_image: &DynamicImage) -> RgbaImage {
	let tile_size = input.tile_size.get();

	let mut grid: Grid<(u32, u32, u32, u32)> = Grid::new_fn(Size::new(tile_size, tile_size), |_| (0, 0, 0, 0));

	let mut image = RgbaImage::new(tile_size, tile_size);

	if references.is_empty() {
		return image;
	}

	for reference in references {
		let image = tile_to_image(input, reference, tileset_image);

		grid.enumerate_mut().for_each(|(Coord {x, y}, (r_v, g_v, b_v, a_v))| {
			let &image::Rgba([r, g, b, a]) = image.get_pixel(x as u32, y as u32);

			let (r, g, b, a) = (r as u32, g as u32, b as u32, a as u32);

			*r_v += r * a;
			*g_v += g * a;
			*b_v += b * a;
			*a_v = (*a_v).max(a);
		});
	}

	grid.enumerate().for_each(|(Coord {x, y}, (r, g, b, a))| {
		let div = references.len() as u32 * 255;
		image.put_pixel(x as u32, y as u32, image::Rgba([(r / div) as u8, (g / div) as u8, (b / div) as u8, *a as u8]))
	});

	image
}

#[test]
fn test_determinism() {
	for x in 1 ..= 2 {
		run(Opt {
			input_config: "examples/simple.ron".into(),
			subcommand: Subcommand::GenerateTilemap {
				generation_options: GenerationOptions {
					map_size: MapSize {
						width: NonZeroU16::new(10).unwrap(),
						height: NonZeroU16::new(10).unwrap()
					},
					attempts: NonZeroUsize::new(1000).unwrap(),
					contradiction_image_path: None,
					wrap: Wrap::None,
					seed: Some(1)
				},
				path: format!("tests/deterministic_{}.ron", x).into(),
			}
		}).unwrap();
	}

	assert_eq!(
		std::fs::read("tests/deterministic_1.ron").unwrap(),
		std::fs::read("tests/deterministic_2.ron").unwrap(),
	)
}
