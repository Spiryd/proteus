use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{Duration, interval};

pub struct RateLimiter {
    semaphore: Arc<Semaphore>,
}

impl RateLimiter {
    pub fn new(requests_per_minute: u32) -> Self {
        assert!(requests_per_minute > 0, "requests_per_minute must be > 0");
        let semaphore = Arc::new(Semaphore::new(requests_per_minute as usize));
        let sem = semaphore.clone();
        let refill_ms = 60_000u64 / u64::from(requests_per_minute);
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_millis(refill_ms));
            ticker.tick().await;
            loop {
                ticker.tick().await;
                if sem.available_permits() < requests_per_minute as usize {
                    sem.add_permits(1);
                }
            }
        });
        Self { semaphore }
    }

    pub async fn acquire(&self) {
        let permit = self
            .semaphore
            .acquire()
            .await
            .expect("rate limiter semaphore unexpectedly closed");
        permit.forget();
    }
}
