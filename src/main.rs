use image::*;
use direction::CardinalDirection as Direction;

#[derive(Copy, Clone)]
enum Image {
    Wall, Floor, Ground, RockyGround,
    StairsR2L,
}

impl Image {
    fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::Wall,
            1 => Self::Floor,
            2 => Self::Ground,
            3 => Self::RockyGround,
            4 => Self::StairsR2L,
            _ => panic!()
        }
    }

    fn to_u32(self) -> u32 {
        match self {
            Self::Wall => 0,
            Self::Floor => 1,
            Self::Ground => 2,
            Self::RockyGround => 3,
            Self::StairsR2L => 4,
        }
    }

    fn coords(self) -> (u32, u32) {
        match self {
            Self::Wall => (1, 1),
            Self::Floor => (2, 1),
            Self::Ground => (3, 1),
            Self::RockyGround => (4, 1),

            Self::StairsR2L => (1, 2),
        }
    }

    fn allowed_neighbours(self, direction: Direction) -> Vec<Image> {
        match self {
            this @ Self::Wall => match direction {
                _ => vec![Self::Floor],
            }
            this @ Self::Floor => match direction {
                _ => vec![Self::Wall]
            },
            Self::Ground => vec![Self::Wall],
            Self::RockyGround => vec![Self::RockyGround, Self::Wall],
            Self::StairsR2L => match direction {
                Direction::East => vec![Self::Floor],
                Direction::West => vec![Self::Wall],
                Direction::North | Direction::South => vec![Self::Floor, Self::Wall],
            }
        }
    }
}

fn main() {
    let xxx = image::open("tiles.png").unwrap();


    let mut context = wfc::Context::new();
    let size = wfc::Size::new_u16(10, 10);
    let retry = wfc::retry::Forever;

    let mut pattern_table = wfc::PatternTable::from_vec(
        [Image::Wall, Image::Floor]
            .iter()
            .map(|image| wfc::PatternDescription {
                weight: None,
                allowed_neighbours: direction::CardinalDirectionTable::new_array([
                    image.allowed_neighbours(Direction::North).iter().map(|image| image.to_u32()).collect(),
                    image.allowed_neighbours(Direction::East).iter().map(|image| image.to_u32()).collect(),
                    image.allowed_neighbours(Direction::South).iter().map(|image| image.to_u32()).collect(),
                    image.allowed_neighbours(Direction::West).iter().map(|image| image.to_u32()).collect(),
                ])
            })
            .collect()
    );

    let mut rng = rand::thread_rng();
    let global_stats = wfc::GlobalStats::new(pattern_table);

    let mut rb = wfc::RunOwn::new_forbid(
        wfc::Size::new_u16(10, 10),
        //&mut context, &mut wave,
        &global_stats,
        Corner,
        &mut rng
    );

    let wave = rb.collapse_retrying(retry, &mut rng);

    let image = {
        let size = wave.grid().size();
        let mut rgba_image = image::RgbaImage::new(size.width() * 16, size.height() * 16);
        wave.grid().enumerate().for_each(|(wfc::Coord { mut x, mut y }, cell)| {
            x *= 16;
            y *= 16;

            println!("{} {} {:?}", x, y, cell);

            /*/let colour = match cell.chosen_pattern_id() {
                Ok(pattern_id) => {
                    *self.overlapping_patterns.pattern_top_left_value(pattern_id)
                }
                Err(_) => self.empty_colour,
            };*/
            let image = match cell.chosen_pattern_id() {
                Ok(cell) =>  Image::from_u32(cell),
                _ => Image::RockyGround,
            };
            let (coord_x, coord_y) = image.coords();
            let image = xxx.view(coord_x * 16, coord_y * 16, 16, 16);

            image::imageops::overlay(
                &mut rgba_image,
                &image,
                x as u32, y as u32,
            )
        });
        DynamicImage::ImageRgba8(rgba_image)
    };

    image.save("out.png").unwrap();
}

#[derive(Copy, Clone)]
struct Corner;

impl wfc::ForbidPattern for Corner {
    fn forbid<W: wfc::Wrap, R: rand::Rng>(
        &mut self, 
        fi: &mut wfc::ForbidInterface<W>,
        rng: &mut R,
    ) {
        fi.forbid_all_patterns_except(wfc::Coord::new(0, 0), Image::Wall.to_u32(), rng);

    }
}
