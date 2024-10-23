use itertools::Itertools;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TernaryCode {
    Zero,
    One,
    Two,
}

pub const LABEL_20: [f32; 3] = [10. / 120., 15. / 120., 20. / 120.];
pub const LABEL_HD: [f32; 3] = [10. / 165., 20. / 165., 30. / 165.];

impl TernaryCode {
    #[allow(dead_code)]
    pub(crate) fn to_code(val: usize) -> Vec<TernaryCode> {
        let mut v = val;
        let mut res = Vec::<TernaryCode>::new();
        while v > 0 {
            let code = match v % 3 {
                0 => TernaryCode::Zero,
                1 => TernaryCode::One,
                2 => TernaryCode::Two,
                _ => unreachable!(),
            };
            res.push(code);
            v /= 3;
        }
        for _ in res.len()..5 {
            res.push(TernaryCode::Zero);
        }
        res.reverse();
        res
    }

    pub(crate) fn to_int(codes: &[TernaryCode]) -> usize {
        let mut codes = codes.iter().copied().collect_vec();
        codes.reverse();
        let mut fact = 1_usize;
        let mut val = 0_usize;
        codes.iter().for_each(|&code| {
            match code {
                TernaryCode::Zero => (),
                TernaryCode::One => val += fact,
                TernaryCode::Two => val += 2 * fact,
            };
            fact *= 3;
        });
        val
    }
}

pub(crate) fn decode_fiducial(norm_radius_list: &[f32], label: &[f32]) -> Vec<TernaryCode> {
    norm_radius_list
        .iter()
        .map(|x| {
            let diff = label
                .iter()
                .map(|y| (x - y).abs())
                .enumerate()
                .fold(
                    (10, 1.0), //max value for val is < 1.0
                    |(idx_min, val_min), (idx, val)| {
                        if val_min < val {
                            (idx_min, val_min)
                        } else {
                            (idx, val)
                        }
                    },
                )
                .0;
            match diff {
                0 => TernaryCode::Zero,
                1 => TernaryCode::One,
                2 => TernaryCode::Two,
                _ => unreachable!("Could not encode properly"),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::encoding::{decode_fiducial, TernaryCode};

    use super::LABEL_HD;

    ///Currently only support encoding of 5 bit ternary code
    //function to encode a number to three concentric rings
    ///word: a ternary sequence
    //hi_width width represents 2 of the ternary code
    //mid_width width represents 1 of the ternary code
    //low_width width represents 0 of the ternary code
    //width_options contains the above three width
    ///outer_radius  largest radius

    pub(crate) fn ternary_encode(
        word: Vec<TernaryCode>,
        outer_radius: f32,
        width_options: [f32; 3],
    ) -> Vec<f32> {
        let mut last_radius = outer_radius;
        let mut radius_list = vec![outer_radius];
        word.iter().for_each(|code| {
            match code {
                TernaryCode::Two => last_radius -= width_options[0],
                TernaryCode::One => last_radius -= width_options[1],
                TernaryCode::Zero => last_radius -= width_options[2],
            }
            radius_list.push(last_radius);
        });
        radius_list
    }

    #[test]
    fn test_ternary_code() {
        let word = vec![
            TernaryCode::Zero,
            TernaryCode::Zero,
            TernaryCode::Zero,
            TernaryCode::One,
            TernaryCode::Two,
        ];

        let outer_radius = 120.0;
        let width_options = [10.0, 15., 25.0];
        let res = ternary_encode(word, outer_radius, width_options);
        assert!(res.iter().eq([120.0, 95.0, 70.0, 45.0, 30.0, 20.0].iter()));
    }

    #[test]
    fn test_decode() {
        let norm_radius_list = [
            [0.18144378, 0.1801021, 0.12617005, 0.17777021, 0.12861128],
            [0.18297406, 0.1807139, 0.12231276, 0.11518198, 0.08002022],
            [0.18208434, 0.17396745, 0.0758446, 0.1153341, 0.12506016],
            [0.18269047, 0.17378184, 0.06903974, 0.05992889, 0.07268634],
            [0.11693681, 0.07536138, 0.11220046, 0.1271825, 0.17819881],
        ];

        let label = LABEL_HD;
        let codes: Vec<_> = norm_radius_list
            .iter()
            .map(|x| decode_fiducial(x, &label))
            .collect();
        let vals: Vec<_> = codes.iter().map(|x| TernaryCode::to_int(x)).collect();
        assert_eq!(vals, [232, 228, 220, 216, 95]);
    }
    #[test]
    fn test_gencode() {
        let width_options = [10.0, 15., 25.0];
        let val = 140;
        let word = TernaryCode::to_code(val);
        let encoded = ternary_encode(word, 120., width_options);
        assert!(encoded
            .iter()
            .eq([120.0, 105.0, 95.0, 70.0, 55.0, 45.0].iter()));
    }
}
