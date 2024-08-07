import asyncio
from crawlee.playwright_crawler import PlaywrightCrawler, PlaywrightCrawlingContext
from crawlee.autoscaling.autoscaled_pool import ConcurrencySettings
from datetime import timedelta
import time
import random
import os
import shutil

def random_delay(min_delay_ms=100, max_delay_ms=300):
    time.sleep(random.uniform(min_delay_ms, max_delay_ms) / 1000)


async def random_scroll_to_bottom(page):
    # Get the height of the viewport
    viewport_height = await page.evaluate("window.innerHeight")

    # Get the total height of the page's content
    content_height = await page.evaluate("document.body.scrollHeight")

    # Calculate the number of scrolls needed
    num_scrolls = random.randint(5, 10)  # Random number of scrolls between 5 and 10

    for _ in range(num_scrolls):
        # Calculate a random scroll position within the page height
        scroll_position = random.randint(0, content_height - viewport_height)

        random_delay()
        
        # Scroll to the random position
        await page.evaluate(f"window.scrollTo(0, {scroll_position});")
        
        # Introduce a random pause between scrolls
        await asyncio.sleep(random.uniform(0.5, 2.0))

    # Finally, scroll to the bottom
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
    await asyncio.sleep(random.uniform(1.0, 3.0))  # Random pause at the end


async def scroll_and_wait_for_jobs(page, max_scrolls=50):
    scroll_count = 0
    no_change_count = 0
    max_no_change = 10
    job_selector = "ul.jobs-search__results-list li"

    while scroll_count < max_scrolls and no_change_count < max_no_change:
        # Get the current number of job posts
        current_job_count = len(await page.query_selector_all(job_selector))
        print(f"Current job count: {current_job_count}")

        # Scroll down to the bottom
        await random_scroll_to_bottom(page)  # or scroll_to_bottom

        # Wait for new job posts to load
        await page.wait_for_timeout(9000)  # Adjust the timeout as needed

        # Click the "See More Jobs" button if it is visible
        is_visible = await page.evaluate('''
            () => {
                const button = document.querySelector('button[aria-label="See more jobs"]');
                if (button) {
                    const style = window.getComputedStyle(button);
                    return style.display === 'inline-block';
                }
                return false;
            }
        ''')

        random_delay()

        if is_visible:
            print("The 'See More Jobs' button is now visible.")
            await page.click('button[aria-label="See more jobs"]')
            await page.wait_for_timeout(5000)  # Adjust the timeout as needed

        # Check if the number of job posts has increased
        new_job_count = len(await page.query_selector_all(job_selector))
        if new_job_count > current_job_count:
            scroll_count += 1  # Increment scroll count if new jobs have loaded
            no_change_count = 0  # Reset no change count
        else:
            no_change_count += 1  # Increment no change count if no new jobs have loaded

        print(f"Scrolled {scroll_count} times. No change count: {no_change_count}")

    print(f"Total scrolls performed: {scroll_count}")

def generate_linkedin_urls(cities):
    base_url = "www.jobsearch.com"
    urls = []
    
    for city in cities:
        city_formatted = city.replace(" ", "%20")
        url = f"{base_url}{city_formatted}"
        urls.append(url)
    
    return urls

async def main() -> None:
        #Initialize the concurrency settings
    concurrency_settings = ConcurrencySettings(
        min_concurrency=5,
        max_concurrency=8,
        max_tasks_per_minute=3,
        desired_concurrency=5
    )

    crawler = PlaywrightCrawler(browser_type='webkit',
                                request_handler_timeout=timedelta(seconds=10000),
                                concurrency_settings=concurrency_settings)

    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        context.log.info(f'Processing {context.request.url} ...')

        # Navigate to the LinkedIn job search page
        await context.page.goto(context.request.url, timeout=0)

        # Try scrolling down
        await scroll_and_wait_for_jobs(context.page)

        # Initialize a list to hold job data
        jobs = []

        # Extract job data
        job_results_container = await context.page.query_selector('ul.jobs-search__results-list')
        
        if job_results_container:
            # If the container exists, count the job post elements within it
            job_cards = await job_results_container.query_selector_all('li')
            print(f"There are {len(job_cards)} jobs found")

            for job_card in job_cards:
                job_title = await job_card.query_selector('h3.base-search-card__title')
                company_name = await job_card.query_selector('a.hidden-nested-link')
                location = await job_card.query_selector('span.job-search-card__location')
                element_handle = await job_card.query_selector('[class*="base-card__full-link"]')
                
                if element_handle:
                    job_link = await element_handle.get_attribute('href')
                else:
                    job_link = "NA"

                jobs.append({
                    'title': await job_title.inner_text() if job_title else 'N/A',
                    'company': await company_name.inner_text() if company_name else 'N/A',
                    'location': await location.inner_text() if location else 'N/A',
                    'url': job_link
                })

        # Log the number of jobs collected
        context.log.info(f'Collected {len(jobs)} jobs.')

        # Push the collected job data to the dataset
        for job in jobs:
            await context.push_data(job)

    # Run the crawler with the LinkedIn job search URL
    await crawler.run(
        generate_linkedin_urls(["city1","city2","city3"])
    )

asyncio.run(main())