use rand::{
    distributions::uniform::SampleUniform,
    rngs::StdRng,
    Rng,
};
use rand_distr::{Distribution, Standard, Normal, StandardNormal};
use std::{
    // marker::PhantomData,
    ops::AddAssign
};

/// A generator
pub trait Generator<R = StdRng>: Sized
{
    type Item: Sized;
    // Provided method
    fn gen(&self, rng: &mut R) -> Self::Item;
       // where
       //  Standard: Distribution<Self::Item>,
       //  R: Rng;// + ?Sized;


    /// Create an iterator of the given generator.
    fn into_iter(self, rng: &mut R) -> impl Iterator<Item = Self::Item>
    where
        Self: Sized,
    {
        std::iter::repeat_with(move || self.gen(rng))
    }

    /// Turn this generator into a mutator via the given closure.
    fn into_mutator<F>(self, f: F) -> impl Mutator<R, Self::Item>
    // fn into_mutator<F,M>(self, f: F) -> impl Mutator<R, M>
    where
        Self: Sized,
        F: Fn(Self::Item, &mut Self::Item),
    {
        move |genome: &mut Self::Item, rng: &mut R| {
            let generated = self.gen(rng);
            f(generated, genome);
            1
        }
    }
}

// struct FnMarker<G>(PhantomData<G>);

/// A mutator for type `G` in association with type `R`, which is typically a
/// random number generator.
pub trait Mutator<R = StdRng, Marker = ()> {
    type Item;
    /// Mutate the `genome` returning the number of mutations that occurred.
    fn mutate(&self, genome: &mut Self::Item, rng: &mut R) -> u32;
}

// impl<F, R, G> Mutator<R, FnMarker<G>> for F
/// We use `G` here as a marker attribute. `Self::Item` defines the genome type.
impl<F, R, G> Mutator<R, G> for F
where
    F: Fn(&mut G, &mut R) -> u32,
{
    type Item = G;
    fn mutate(&self, value: &mut Self::Item, rng: &mut R) -> u32 {
        self(value, rng)
    }
}
/// Generate a normal distribution with the given $mean$ and $stddev$.
pub fn normal_generator<T, R>(mean: T, stddev: T) -> Option<impl Generator<R, Item = T>>
where
    T: PartialOrd + Copy + rand_distr::num_traits::Float,
    StandardNormal: Distribution<T>,
    R: Rng,
{
    Normal::new(mean, stddev)
        .map(|n| move |rng: &mut R| n.sample(rng))
        .ok()
}

/// Generate a uniform distribution $U ~ [min, max)$.
pub fn uniform_generator<T, R>(min: T, max: T) -> impl Generator<R, Item = T>
where
    T: SampleUniform + PartialOrd + Copy,
    Standard: Distribution<T>,
    R: Rng,

{
    move |rng: &mut R| rng.gen_range(min..max)
}

impl<F, R, G> Generator<R> for F
where
    F: Fn(&mut R) -> G,
{
    type Item = G;

    fn gen(&self, rng: &mut R) -> G {
        self(rng)
    }
}


/// Add a value drawn from a uniform distribution $U ~ [min, max)$.
pub fn uniform_mutator<T, R>(min: T, max: T) -> impl Mutator<R, T>
where
    T: SampleUniform + PartialOrd + Copy + AddAssign<T>,
    R: Rng,
{
    move |value: &mut T, rng: &mut R| {
        *value += rng.gen_range(min..max);
        1
    }
}


// Add a value drawn from a normal distribution with the given $mean$ and
// $stddev$.
pub fn normal_mutator<T, R>(mean: T, stddev: T) -> Option<impl Mutator<R, T>>
where
    T: PartialOrd + Copy + rand_distr::num_traits::Float + AddAssign<T>,
    StandardNormal: Distribution<T>,
    R: Rng,
{
    normal_generator(mean, stddev)
        .map(|generator|
             generator.into_mutator(|generated, mutated| *mutated += generated))
}

/// Random number generator extensions.
pub trait RngExt {
    /// Return a probability $p \in (0, 1)$.
    fn prob(&mut self) -> f32;
    /// Return true with a probability $p \in [0, 1]$.
    fn with_prob(&mut self, p: f32) -> bool;
}

impl<R: Rng> RngExt for R {
    fn prob(&mut self) -> f32 {
        self.sample(rand::distributions::Open01)
    }
    fn with_prob(&mut self, p: f32) -> bool {
        p > self.prob()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_generator() {
        let mut rng = rand::thread_rng();
        let v: Vec<f32> = RngExt::prob.into_iter(&mut rng).take(2).collect();
        assert!(v[0] > 0.0 && v[0] < 1.0);
        assert!(v[1] > 0.0 && v[1] < 1.0);
    }

    #[test]
    fn test_uniform_generator() {
        let mut rng = rand::thread_rng();
        let g = uniform_generator(0u32, 100u32);
        // let x = g(&mut rng);
        let x: u32 = g.gen(&mut rng);
        assert!(x > 0 && x < 100);
    }
}
