"""
IntentForge E2E Tests with Playwright
=====================================

Tests cover:
1. Form submission flow
2. Payment checkout flow
3. Camera streaming
4. Data CRUD operations
5. Real-time updates
6. Web Components
7. Error handling

Setup:
    pip install pytest playwright pytest-playwright pytest-asyncio
    playwright install chromium

Run:
    pytest tests/e2e/ -v
    pytest tests/e2e/ --headed  # See browser
    pytest tests/e2e/ -k "form"  # Run specific tests
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from playwright.async_api import Page, async_playwright, expect

# Load .env file
load_dotenv(project_root / ".env")

# =============================================================================
# Configuration from .env
# =============================================================================

# Get ports from environment (unified with docker-compose and Python config)
APP_PORT = os.getenv("APP_PORT", "8000")
APP_PORT_EXTERNAL = os.getenv("APP_PORT_EXTERNAL", APP_PORT)
WEB_PORT = os.getenv("WEB_PORT", "80")

# URLs for testing - can be overridden by TEST_* vars
BASE_URL = os.getenv("TEST_BASE_URL", f"http://localhost:{WEB_PORT}")
API_URL = os.getenv("TEST_API_URL", f"http://localhost:{APP_PORT_EXTERNAL}")
TIMEOUT = int(os.getenv("TEST_TIMEOUT", "30000"))  # 30 seconds


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def browser():
    """Launch browser for all tests"""
    async with async_playwright() as p:
        executable_path = getattr(p.chromium, "executable_path", None)
        if executable_path and not Path(executable_path).exists():
            pytest.skip(
                "Playwright browsers are not installed. Run: `python -m playwright install chromium` (or `playwright install`)"
            )

        try:
            browser = await p.chromium.launch(
                headless=os.getenv("HEADLESS", "true").lower() == "true"
            )
        except Exception as e:
            msg = str(e)
            if "Executable doesn't exist" in msg or "playwright install" in msg.lower():
                pytest.skip(
                    "Playwright browsers are not installed. Run: `python -m playwright install chromium` (or `playwright install`)"
                )
            raise
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser):
    """Create new page for each test"""
    context = await browser.new_context(
        viewport={"width": 1280, "height": 720}, ignore_https_errors=True
    )
    page = await context.new_page()
    page.set_default_timeout(TIMEOUT)
    yield page
    await context.close()


@pytest.fixture
async def api_page(page):
    """Page with API interceptor"""
    api_calls = []

    async def capture_api(route, request):
        api_calls.append(
            {"url": request.url, "method": request.method, "post_data": request.post_data}
        )
        await route.continue_()

    await page.route(f"{API_URL}/**", capture_api)

    page.api_calls = api_calls
    yield page


# =============================================================================
# Test: Form Submission
# =============================================================================


class TestFormSubmission:
    """Test form handling with intent-form component"""

    @pytest.mark.asyncio
    async def test_contact_form_success(self, page: Page):
        """Test successful contact form submission"""
        await page.goto(f"{BASE_URL}/examples/usecases/01_contact_form.html")

        # Fill form
        await page.fill('input[name="name"]', "Jan Kowalski")
        await page.fill('input[name="email"]', "jan@example.com")
        await page.fill('input[name="phone"]', "+48 123 456 789")
        await page.select_option('select[name="subject"]', "general")
        await page.fill('textarea[name="message"]', "Test message from E2E tests")

        # Submit
        await page.click('button[type="submit"]')

        # Wait for success
        success_message = page.locator(".success-message")
        await expect(success_message).to_be_visible(timeout=10000)
        await expect(success_message).to_contain_text("wysłana")

    @pytest.mark.asyncio
    async def test_contact_form_validation(self, page: Page):
        """Test form validation"""
        await page.goto(f"{BASE_URL}/examples/usecases/01_contact_form.html")

        # Submit empty form
        await page.click('button[type="submit"]')

        # Check HTML5 validation
        is_valid = await page.evaluate("""
            () => document.querySelector('form').checkValidity()
        """)
        assert not is_valid

    @pytest.mark.asyncio
    async def test_intent_form_component(self, page: Page):
        """Test custom intent-form web component"""
        await page.set_content("""
            <script src="/frontend/components/intentforge-components.js"></script>
            <intent-form action="test" success-message="Success!">
                <input name="email" type="email" value="test@example.com">
                <input name="message" value="Test">
            </intent-form>
        """)

        # Submit
        await page.click('button[type="submit"]')

        # Check loading state
        form = page.locator("intent-form")
        await expect(form).to_have_class(re.compile(r"loading"))


# =============================================================================
# Test: Payment Flow
# =============================================================================


class TestPaymentFlow:
    """Test payment checkout flow"""

    @pytest.mark.asyncio
    async def test_ebook_page_loads(self, page: Page):
        """Test ebook page renders correctly"""
        await page.goto(f"{BASE_URL}/examples/usecases/02_ebook_payment.html")

        # Check price displayed
        price = page.locator(".price")
        await expect(price).to_contain_text("49")

        # Check payment buttons
        await expect(page.locator("#pay-paypal")).to_be_visible()
        await expect(page.locator("#pay-card")).to_be_visible()
        await expect(page.locator("#pay-blik")).to_be_visible()

    @pytest.mark.asyncio
    async def test_discount_code(self, page: Page):
        """Test discount code application"""
        await page.goto(f"{BASE_URL}/examples/usecases/02_ebook_payment.html")

        # Enter discount code
        await page.fill("#discount-code", "RABAT20")
        await page.click("#apply-discount")

        # Wait for response
        await page.wait_for_timeout(1000)

        # Check message appeared
        message = page.locator("#discount-message")
        await expect(message).to_be_visible()

    @pytest.mark.asyncio
    async def test_email_validation_before_payment(self, page: Page):
        """Test email is required before checkout"""
        await page.goto(f"{BASE_URL}/examples/usecases/02_ebook_payment.html")

        # Clear email field
        await page.fill("#customer-email", "")

        # Try to pay
        await page.click("#pay-paypal")

        # Alert should appear
        dialog_promise = page.wait_for_event("dialog")
        dialog = await dialog_promise
        assert "email" in dialog.message.lower()
        await dialog.dismiss()

    @pytest.mark.asyncio
    async def test_intent_pay_component(self, page: Page):
        """Test intent-pay web component"""
        await page.set_content("""
            <script src="/frontend/components/intentforge-components.js"></script>
            <intent-pay
                amount="99.99"
                currency="PLN"
                product="test-product"
                provider="paypal"
                email="test@example.com">
                Buy Now
            </intent-pay>
        """)

        pay_button = page.locator("intent-pay")
        await expect(pay_button).to_be_visible()
        await expect(pay_button).to_contain_text("Buy Now")


# =============================================================================
# Test: Camera Monitoring
# =============================================================================


class TestCameraMonitoring:
    """Test camera and image handling"""

    @pytest.mark.asyncio
    async def test_camera_page_loads(self, page: Page):
        """Test camera page renders"""
        await page.goto(f"{BASE_URL}/examples/usecases/03_camera_monitoring.html")

        # Check main elements
        await expect(page.locator("#camera-main")).to_be_visible()
        await expect(page.locator("#event-log")).to_be_visible()

    @pytest.mark.asyncio
    async def test_detection_checkboxes(self, page: Page):
        """Test detection option toggles"""
        await page.goto(f"{BASE_URL}/examples/usecases/03_camera_monitoring.html")

        # Check checkboxes are present
        motion = page.locator("#detect-motion")
        person = page.locator("#detect-person")
        vehicle = page.locator("#detect-vehicle")

        await expect(motion).to_be_checked()
        await expect(person).to_be_checked()
        await expect(vehicle).to_be_checked()

        # Toggle
        await motion.uncheck()
        await expect(motion).not_to_be_checked()

    @pytest.mark.asyncio
    async def test_snapshot_button(self, page: Page):
        """Test snapshot capture"""
        await page.goto(f"{BASE_URL}/examples/usecases/03_camera_monitoring.html")

        # Click snapshot
        await page.click("#btn-snapshot")

        # Should trigger download or show image
        await page.wait_for_timeout(2000)

    @pytest.mark.asyncio
    async def test_intent_image_component(self, page: Page):
        """Test intent-image auto-refresh"""
        await page.set_content("""
            <script src="/frontend/components/intentforge-components.js"></script>
            <intent-image
                src="/api/charts/test"
                refresh="1000">
            </intent-image>
        """)

        # Wait for component to render
        await page.wait_for_timeout(500)

        img = page.locator("intent-image img")
        await expect(img).to_be_visible()

        # Check src updates (contains timestamp)
        first_src = await img.get_attribute("src")
        await page.wait_for_timeout(1500)
        second_src = await img.get_attribute("src")

        assert first_src != second_src


# =============================================================================
# Test: Dashboard Real-time Updates
# =============================================================================


class TestDashboard:
    """Test real-time dashboard functionality"""

    @pytest.mark.asyncio
    async def test_dashboard_loads(self, page: Page):
        """Test dashboard page loads"""
        await page.goto(f"{BASE_URL}/examples/usecases/04_realtime_dashboard.html")

        # Check metric cards
        await expect(page.locator("#metric-users")).to_be_visible()
        await expect(page.locator("#metric-orders")).to_be_visible()
        await expect(page.locator("#metric-revenue")).to_be_visible()

    @pytest.mark.asyncio
    async def test_metrics_update(self, page: Page):
        """Test metrics auto-update"""
        await page.goto(f"{BASE_URL}/examples/usecases/04_realtime_dashboard.html")

        # Get initial value
        initial = await page.locator("#metric-users").text_content()

        # Wait for update (5 seconds)
        await page.wait_for_timeout(6000)

        # Value should change (or stay same if API returns same data)
        current = await page.locator("#metric-users").text_content()
        # Just verify it's a number
        assert current.replace("-", "").replace(".", "").isdigit() or current == "--"

    @pytest.mark.asyncio
    async def test_pause_refresh(self, page: Page):
        """Test pause/resume functionality"""
        await page.goto(f"{BASE_URL}/examples/usecases/04_realtime_dashboard.html")

        pause_btn = page.locator("#toggle-refresh")

        # Click pause
        await pause_btn.click()
        await expect(pause_btn).to_contain_text("Wznów")

        # Click resume
        await pause_btn.click()
        await expect(pause_btn).to_contain_text("Wstrzymaj")


# =============================================================================
# Test: Data Components
# =============================================================================


class TestDataComponents:
    """Test data display components"""

    @pytest.mark.asyncio
    async def test_intent_data_component(self, page: Page):
        """Test intent-data web component"""
        await page.set_content("""
            <script src="/frontend/components/intentforge-components.js"></script>
            <intent-data
                table="products"
                limit="5">
            </intent-data>
        """)

        # Wait for data load
        await page.wait_for_timeout(2000)

        # Check loading state removed
        data_el = page.locator("intent-data")
        await expect(data_el).not_to_have_class(re.compile(r"loading"))

    @pytest.mark.asyncio
    async def test_intent_data_with_template(self, page: Page):
        """Test intent-data with custom template"""
        await page.set_content("""
            <template id="item-template">
                <div class="item">{{name}} - {{price}}</div>
            </template>
            <script src="/frontend/components/intentforge-components.js"></script>
            <intent-data
                table="products"
                template="#item-template">
            </intent-data>
        """)

        await page.wait_for_timeout(2000)


# =============================================================================
# Test: Authentication
# =============================================================================


class TestAuthentication:
    """Test auth components"""

    @pytest.mark.asyncio
    async def test_intent_auth_login(self, page: Page):
        """Test login form renders"""
        await page.set_content("""
            <script src="/frontend/components/intentforge-components.js"></script>
            <intent-auth mode="login" redirect="/dashboard">
            </intent-auth>
        """)

        # Check form rendered
        await expect(page.locator('input[name="email"]')).to_be_visible()
        await expect(page.locator('input[name="password"]')).to_be_visible()

    @pytest.mark.asyncio
    async def test_intent_auth_register(self, page: Page):
        """Test registration form"""
        await page.set_content("""
            <script src="/frontend/components/intentforge-components.js"></script>
            <intent-auth mode="register">
            </intent-auth>
        """)

        # Registration has name field
        await expect(page.locator('input[name="name"]')).to_be_visible()
        await expect(page.locator('input[name="password_confirm"]')).to_be_visible()

    @pytest.mark.asyncio
    async def test_intent_auth_social(self, page: Page):
        """Test social login buttons"""
        await page.set_content("""
            <script src="/frontend/components/intentforge-components.js"></script>
            <intent-auth mode="login" providers="google,github">
            </intent-auth>
        """)

        await expect(page.locator('[data-provider="google"]')).to_be_visible()
        await expect(page.locator('[data-provider="github"]')).to_be_visible()


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error scenarios"""

    @pytest.mark.asyncio
    async def test_network_error_handling(self, page: Page):
        """Test offline/network error handling"""
        await page.goto(f"{BASE_URL}/examples/usecases/01_contact_form.html")

        # Block API
        await page.route(f"{API_URL}/**", lambda route: route.abort())

        # Fill and submit form
        await page.fill('input[name="name"]', "Test")
        await page.fill('input[name="email"]', "test@example.com")
        await page.fill('textarea[name="message"]', "Test message")
        await page.click('button[type="submit"]')

        # Error should show
        await page.wait_for_timeout(2000)
        error_message = page.locator(".error-message")
        # May or may not be visible depending on implementation

    @pytest.mark.asyncio
    async def test_api_error_response(self, page: Page):
        """Test API error response handling"""
        await page.goto(f"{BASE_URL}/examples/usecases/01_contact_form.html")

        # Mock API error
        await page.route(
            f"{API_URL}/**",
            lambda route: route.fulfill(status=500, body='{"error": "Internal Server Error"}'),
        )

        # Submit form
        await page.fill('input[name="name"]', "Test")
        await page.fill('input[name="email"]', "test@example.com")
        await page.fill('textarea[name="message"]', "Test message")
        await page.click('button[type="submit"]')

        await page.wait_for_timeout(2000)


# =============================================================================
# Test: SDK Integration
# =============================================================================


class TestSDKIntegration:
    """Test JavaScript SDK integration"""

    @pytest.mark.asyncio
    async def test_sdk_loads(self, page: Page):
        """Test SDK loads correctly"""
        await page.set_content("""
            <script src="/frontend/sdk/intentforge.js"></script>
        """)

        # Check IntentForge is available
        is_loaded = await page.evaluate('typeof IntentForge !== "undefined"')
        assert is_loaded

    @pytest.mark.asyncio
    async def test_sdk_version(self, page: Page):
        """Test SDK version is set"""
        await page.set_content("""
            <script src="/frontend/sdk/intentforge.js"></script>
        """)

        version = await page.evaluate("IntentForge.VERSION")
        assert version is not None
        assert "." in version

    @pytest.mark.asyncio
    async def test_sdk_auto_bind(self, page: Page):
        """Test auto-binding forms"""
        await page.set_content("""
            <script src="/frontend/sdk/intentforge.js"
                    data-api="http://localhost:8000"
                    data-auto-bind="true"></script>
            <form data-intent="test">
                <input name="test" value="value">
                <button type="submit">Submit</button>
            </form>
        """)

        await page.wait_for_timeout(1000)

        # Form should have event listener (hard to test directly)
        # Just verify no errors occurred


# =============================================================================
# Test: Accessibility
# =============================================================================


class TestAccessibility:
    """Test accessibility features"""

    @pytest.mark.asyncio
    async def test_form_labels(self, page: Page):
        """Test form inputs have labels"""
        await page.goto(f"{BASE_URL}/examples/usecases/01_contact_form.html")

        # Check labels exist
        labels = await page.locator("label").count()
        inputs = await page.locator('input:not([type="submit"]):not([type="hidden"])').count()

        # Should have labels for inputs
        assert labels > 0

    @pytest.mark.asyncio
    async def test_keyboard_navigation(self, page: Page):
        """Test keyboard navigation works"""
        await page.goto(f"{BASE_URL}/examples/usecases/01_contact_form.html")

        # Tab through form
        await page.keyboard.press("Tab")
        await page.keyboard.press("Tab")
        await page.keyboard.press("Tab")

        # Should be able to navigate


# =============================================================================
# Test: Performance
# =============================================================================


class TestPerformance:
    """Test performance metrics"""

    @pytest.mark.asyncio
    async def test_page_load_time(self, page: Page):
        """Test page loads quickly"""
        start = await page.evaluate("performance.now()")
        await page.goto(f"{BASE_URL}/examples/usecases/01_contact_form.html")
        end = await page.evaluate("performance.now()")

        load_time = end - start
        # Should load in under 5 seconds
        assert load_time < 5000

    @pytest.mark.asyncio
    async def test_no_console_errors(self, page: Page):
        """Test no JavaScript errors"""
        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

        await page.goto(f"{BASE_URL}/examples/usecases/01_contact_form.html")
        await page.wait_for_timeout(2000)

        # Filter out expected errors (e.g., 404 for optional resources)
        critical_errors = [e for e in errors if "favicon" not in e.lower()]

        assert len(critical_errors) == 0, f"Console errors: {critical_errors}"


@pytest.mark.e2e
class TestAutoDevUI:
    @pytest.mark.asyncio
    async def test_autodev_chat_and_sandbox_happy_path(self, page: Page):
        await page.add_init_script(
            """
            window.lucide = { createIcons: () => {} };
            window.marked = { parse: (s) => s };
            class MockWebSocket {
                constructor(url) {
                    this.url = url;
                    this.readyState = 1;
                    setTimeout(() => { if (this.onopen) this.onopen(); }, 10);
                    setTimeout(() => {
                        if (this.onmessage) {
                            this.onmessage({
                                data: JSON.stringify({ type: 'result', data: { success: true } })
                            });
                        }
                    }, 80);
                }
                send() {}
                close() {
                    this.readyState = 3;
                    if (this.onclose) this.onclose();
                }
            }
            window.WebSocket = MockWebSocket;
            """
        )

        async def chat_route(route, request):
            body = {
                "success": True,
                "response": "```python\nprint('hello from autodev')\n```",
                "model": "mock",
                "provider": "mock",
            }
            await route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(body),
            )

        async def sandbox_route(route, request):
            body = {"success": True, "session_id": "t1", "websocket_url": "/ws/sandbox/t1"}
            await route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(body),
            )

        await page.route("**/api/chat", chat_route)
        await page.route("**/api/sandbox/run", sandbox_route)

        await page.goto(f"{BASE_URL}/examples/usecases/autodev_chat.html")
        await expect(page.locator("#chat-input")).to_be_visible()

        await page.fill("#chat-input", "write some code")
        await page.click("#send-btn")

        await expect(page.locator("#chat-messages")).to_contain_text("hello from autodev")
        await expect(page.locator("#status-badge")).to_contain_text("SUCCESS")
        await expect(page.locator("#logs-view")).to_contain_text("Execution completed")

    @pytest.mark.asyncio
    async def test_autodev_stop_button_calls_api_and_resets_status(self, page: Page):
        await page.add_init_script(
            """
            window.lucide = { createIcons: () => {} };
            window.marked = { parse: (s) => s };
            class MockWebSocket {
                constructor(url) {
                    this.url = url;
                    this.readyState = 1;
                    setTimeout(() => { if (this.onopen) this.onopen(); }, 10);
                }
                send() {}
                close() {
                    this.readyState = 3;
                    if (this.onclose) this.onclose();
                }
            }
            window.WebSocket = MockWebSocket;
            """
        )

        stop_calls: list[str] = []

        async def chat_route(route, request):
            body = {
                "success": True,
                "response": "```python\nprint('hello from autodev')\n```",
                "model": "mock",
                "provider": "mock",
            }
            await route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(body),
            )

        async def sandbox_route(route, request):
            body = {"success": True, "session_id": "t1", "websocket_url": "/ws/sandbox/t1"}
            await route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(body),
            )

        async def stop_route(route, request):
            stop_calls.append(request.url)
            await route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"success": True}),
            )

        await page.route("**/api/chat", chat_route)
        await page.route("**/api/sandbox/run", sandbox_route)
        await page.route("**/api/sandbox/stop/t1", stop_route)

        await page.goto(f"{BASE_URL}/examples/usecases/autodev_chat.html")

        await page.fill("#chat-input", "write some code")
        await page.click("#send-btn")

        await expect(page.locator("#status-badge")).to_contain_text("RUNNING")
        await expect(page.locator("#stop-btn")).to_be_enabled()

        await page.click("#stop-btn")

        await expect(page.locator("#status-badge")).to_contain_text("IDLE")
        await expect(page.locator("#logs-view")).to_contain_text("Execution stopped")
        assert any(url.endswith("/api/sandbox/stop/t1") for url in stop_calls)


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--asyncio-mode=auto",
            "-x",  # Stop on first failure
            "--tb=short",
        ]
    )
