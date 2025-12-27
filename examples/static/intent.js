/**
 * IntentForge Frontend Library
 * Enables declarative intent handling using HTML attributes
 */

class IntentHandler {
    constructor(config = {}) {
        this.apiUrl = config.apiUrl || '/api/intent';
        this.init();
    }

    init() {
        console.log('IntentHandler initialized');
        this.scanAndBind();
    }

    /**
     * Scan DOM for [intent] and [data-intent] elements
     */
    scanAndBind() {
        // Handle Forms
        document.querySelectorAll('form[data-intent]').forEach(form => {
            if (form.dataset.bound) return;
            form.addEventListener('submit', (e) => this.handleFormSubmit(e, form));
            form.dataset.bound = 'true';
            console.log('Bound intention to form:', form.id);
        });

        // Handle Tables (Auto-load data)
        document.querySelectorAll('table[intent]').forEach(table => {
            if (table.dataset.bound) return;
            this.handleTableLoad(table);
            table.dataset.bound = 'true';
            console.log('Bound intention to table:', table.id || 'unnamed');
        });

        // Handle Clickable Elements (Buttons, etc.)
        document.querySelectorAll('[intent]').forEach(el => {
            if (el.dataset.bound || el.tagName === 'TABLE') return; // Skip if bound or is table

            // If it's a submit button inside a form, let the form handler take precedence
            // unless explicit intent is on the button itself.
            // But if the button is submit type, the form submit event will fire.
            // We'll attach to click for generic elements.

            if (el.tagName === 'BUTTON' && el.type === 'submit') {
                // If it's a submit button in a form with data-intent, we might want to skip generic click
                // OR process the button's specific intent in addition?
                // For simplicity: specific intent on button overrides form intent if handled here,
                // BUT preventing default on click stops form submission.

                el.addEventListener('click', (e) => {
                    // If button has its own intent, we process it.
                    // If it's just triggering the form submit, we let the form handler do it to capture all data.
                    e.preventDefault();
                    this.handleElementClick(e, el);
                });
            } else {
                el.addEventListener('click', (e) => this.handleElementClick(e, el));
            }

            el.dataset.bound = 'true';
        });
    }

    /**
     * Handle Table Data Loading
     */
    async handleTableLoad(table) {
        const description = table.getAttribute('intent');
        const context = {};

        // Add any data- attributes as context
        Object.keys(table.dataset).forEach(key => {
            if (key !== 'intent' && key !== 'bound') context[key] = table.dataset[key];
        });

        this.setLoading(table, true);

        try {
            const data = await this.callApi(description, context);
            if (data.success && data.result && data.result.return_value) {
                this.renderTable(table, data.result.return_value);
                this.showSuccess(table, "Data loaded");
            } else {
                console.error("Failed to load table data", data);
                this.showError(table, "Failed to load");
            }
        } catch (e) {
            console.error(e);
            this.showError(table, "Error loading");
        } finally {
            this.setLoading(table, false);
        }
    }

    /**
     * Render data into table
     */
    renderTable(table, data) {
        if (!Array.isArray(data) || data.length === 0) return;

        // Ensure THEAD exists
        let thead = table.querySelector('thead');
        if (!thead) {
            thead = document.createElement('thead');
            table.appendChild(thead);
        }

        // Ensure TBODY exists
        let tbody = table.querySelector('tbody');
        if (!tbody) {
            tbody = document.createElement('tbody');
            table.appendChild(tbody);
        }

        // Render Headers (if empty)
        if (thead.children.length === 0) {
            const headers = Object.keys(data[0]);
            const tr = document.createElement('tr');
            headers.forEach(h => {
                const th = document.createElement('th');
                th.textContent = h.charAt(0).toUpperCase() + h.slice(1);
                tr.appendChild(th);
            });
            thead.appendChild(tr);
        }

        // Render Rows
        tbody.innerHTML = ''; // Clear existing
        data.forEach(row => {
            const tr = document.createElement('tr');
            Object.values(row).forEach(val => {
                const td = document.createElement('td');
                td.textContent = val;
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    /**
     * Unified API Call
     */
    async callApi(description, context) {
        console.log('Processing intent:', description, context);
        const response = await fetch(this.apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                description: description,
                context: context,
                intent_type: "workflow"
            })
        });
        return await response.json();
    }

    /**
     * Handle Form Submission
     */
    async handleFormSubmit(event, form) {
        event.preventDefault();

        const description = form.dataset.intent;
        const resetOnSuccess = form.dataset.intentReset === 'true';

        // Extract data
        const formData = new FormData(form);
        const context = Object.fromEntries(formData.entries());

        // Add specific button intent if available (clicked button)
        // Note: Hard to track which button submitted without 'submit' event submitter support (modern browsers)
        if (event.submitter && event.submitter.getAttribute('intent')) {
            // If the button has an intent, maybe we append it?
            context._trigger_intent = event.submitter.getAttribute('intent');
        }

        await this.processIntent(description, context, form, resetOnSuccess);
    }

    /**
     * Handle Element Click
     */
    async handleElementClick(event, element) {
        event.preventDefault(); // Stop default action (like form submit if it's a button)

        const description = element.getAttribute('intent');

        // Context depends on container. If inside a form, grab form data?
        let context = {};
        const form = element.closest('form');
        if (form) {
            const formData = new FormData(form);
            context = Object.fromEntries(formData.entries());
        }

        // Check if there are specific data attributes to add to context
        Object.keys(element.dataset).forEach(key => {
            if (key !== 'intent' && key !== 'bound') {
                context[key] = element.dataset[key];
            }
        });

        await this.processIntent(description, context, element);
    }

    /**
     * Process Intent via API
     */
    async processIntent(description, context, targetElement, shouldReset = false) {
        // UI Feedback
        const originalText = targetElement.innerHTML || targetElement.innerText;
        this.setLoading(targetElement, true);

        try {
            const data = await this.callApi(description, context);

            if (data.success) {
                this.showSuccess(targetElement, "Success!");
                console.log('Intent Result:', data.result);

                if (shouldReset && targetElement.tagName === 'FORM') {
                    targetElement.reset();
                }
            } else {
                console.error('Intent Failed:', data.message);
                this.showError(targetElement, "Failed");
            }

        } catch (error) {
            console.error('Network/Server Error:', error);
            this.showError(targetElement, "Error");
        } finally {
            // Reset after delay
            setTimeout(() => {
                this.setLoading(targetElement, false);
                // Restore if it was a button/text element
                if (targetElement.tagName === 'BUTTON' || targetElement.tagName === 'A') {
                    targetElement.innerHTML = originalText;
                }
            }, 2000);
        }
    }

    // --- UI Helpers ---

    setLoading(element, isLoading) {
        element.classList.toggle('intent-loading', isLoading);
        if (isLoading) {
            // If it's a button, show spinner
            if (element.tagName === 'BUTTON') {
                element.dataset.originalHtml = element.innerHTML;
                element.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                element.disabled = true;
            }
        } else {
            if (element.tagName === 'BUTTON') {
                element.disabled = false;
            }
        }
    }

    showSuccess(element, message) {
        if (element.tagName === 'BUTTON') {
            element.innerHTML = `✅ ${message}`;
            element.classList.add('btn-success'); // Bootstrap assumption
            setTimeout(() => element.classList.remove('btn-success'), 2000);
        } else {
            // For forms, maybe show a toast or alert
            // Simple generic alert for now if no dedicated status container
            const statusDiv = element.querySelector('.intent-status') || document.createElement('div');
            statusDiv.className = 'intent-status alert alert-success mt-2';
            statusDiv.textContent = message;
            if (!element.contains(statusDiv)) element.appendChild(statusDiv);
            setTimeout(() => statusDiv.remove(), 3000);
        }
    }

    showError(element, message) {
        if (element.tagName === 'BUTTON') {
            element.innerHTML = `❌ ${message}`;
            element.classList.add('btn-danger'); // Bootstrap assumption
            setTimeout(() => element.classList.remove('btn-danger'), 2000);
        } else {
            const statusDiv = element.querySelector('.intent-status') || document.createElement('div');
            statusDiv.className = 'intent-status alert alert-danger mt-2';
            statusDiv.textContent = message;
            if (!element.contains(statusDiv)) element.appendChild(statusDiv);
            setTimeout(() => statusDiv.remove(), 3000);
        }
    }
}

// Auto-initialize
document.addEventListener('DOMContentLoaded', () => {
    window.intentHandler = new IntentHandler();
});
