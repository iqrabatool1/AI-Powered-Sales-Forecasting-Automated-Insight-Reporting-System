import schedule
import time
from automated_pipeline import run_pipeline


def job():
    print("Running scheduled pipeline job...")
    run_pipeline()


# Schedule options (uncomment the one you want):

# Run every day at 8:00 AM
schedule.every().day.at("02:00").do(job)

# Or run every hour (useful for demo/testing purposes)
# schedule.every(1).hours.do(job)

# Or run every 2 minutes (good for quickly demonstrating it works in your viva)
# schedule.every(2).minutes.do(job)


print("Scheduler started. Waiting for scheduled time...")
print("Press Ctrl+C to stop.")

while True:
    schedule.run_pending()
    time.sleep(30)  # check every 30 seconds