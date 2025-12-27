/**
 * IntentForge Web Components
 * ==========================
 * 
 * Ultra-prostsze API - deklaratywne HTML bez pisania JavaScript!
 * 
 * Usage:
 *   <script src="intentforge-components.js"></script>
 *   
 *   <!-- Formularz - zero JS! -->
 *   <intent-form action="contact" success-message="Wysłano!">
 *     <input name="email" type="email" required>
 *     <textarea name="message" required></textarea>
 *   </intent-form>
 *   
 *   <!-- Płatność - jeden tag! -->
 *   <intent-pay amount="49.99" product="ebook" provider="paypal">
 *     Kup teraz
 *   </intent-pay>
 *   
 *   <!-- Auto-odświeżany obraz -->
 *   <intent-image src="/api/chart" refresh="5000"></intent-image>
 *   
 *   <!-- Dane z bazy -->
 *   <intent-data table="products" limit="10" template="#product-card"></intent-data>
 */

(function() {
    'use strict';

    // ==========================================================================
    // Configuration
    // ==========================================================================
    
    const CONFIG = {
        apiUrl: document.currentScript?.dataset?.api || 'http://localhost:8000',
        debug: document.currentScript?.dataset?.debug === 'true'
    };

    // Simple fetch wrapper
    async function api(endpoint, data = {}) {
        const response = await fetch(`${CONFIG.apiUrl}/api/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
            credentials: 'include'
        });
        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        return response.json();
    }

    // ==========================================================================
    // <intent-form> - Formularz bez JS
    // ==========================================================================
    
    class IntentForm extends HTMLElement {
        static get observedAttributes() {
            return ['action', 'success-message', 'error-message', 'redirect'];
        }

        connectedCallback() {
            this.setupForm();
        }

        setupForm() {
            // Wrap content in form if not already
            if (!this.querySelector('form')) {
                const content = this.innerHTML;
                this.innerHTML = `<form>${content}</form>`;
            }

            const form = this.querySelector('form');
            
            // Add submit button if missing
            if (!form.querySelector('[type="submit"]')) {
                const btn = document.createElement('button');
                btn.type = 'submit';
                btn.textContent = this.getAttribute('submit-text') || 'Wyślij';
                form.appendChild(btn);
            }

            // Add status elements
            this.statusEl = document.createElement('div');
            this.statusEl.className = 'intent-status';
            this.statusEl.style.display = 'none';
            form.appendChild(this.statusEl);

            form.addEventListener('submit', (e) => this.handleSubmit(e));
        }

        async handleSubmit(e) {
            e.preventDefault();
            const form = e.target;
            const submitBtn = form.querySelector('[type="submit"]');
            const originalText = submitBtn?.textContent;

            try {
                // Loading state
                this.classList.add('loading');
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.textContent = this.getAttribute('loading-text') || 'Wysyłanie...';
                }

                // Collect form data
                const formData = Object.fromEntries(new FormData(form));
                
                // Send to API
                const result = await api('form', {
                    action: 'submit',
                    form_id: this.getAttribute('action'),
                    data: formData
                });

                // Success
                this.classList.remove('loading');
                this.classList.add('success');
                
                this.showStatus(
                    this.getAttribute('success-message') || '✅ Wysłano pomyślnie!',
                    'success'
                );

                // Reset form
                if (this.getAttribute('reset') !== 'false') {
                    form.reset();
                }

                // Redirect if specified
                const redirect = this.getAttribute('redirect');
                if (redirect) {
                    setTimeout(() => window.location.href = redirect, 1000);
                }

                // Dispatch event
                this.dispatchEvent(new CustomEvent('success', { detail: result }));

            } catch (error) {
                this.classList.remove('loading');
                this.classList.add('error');
                
                this.showStatus(
                    this.getAttribute('error-message') || '❌ Wystąpił błąd. Spróbuj ponownie.',
                    'error'
                );

                this.dispatchEvent(new CustomEvent('error', { detail: error }));

            } finally {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.textContent = originalText;
                }
            }
        }

        showStatus(message, type) {
            this.statusEl.textContent = message;
            this.statusEl.className = `intent-status intent-status-${type}`;
            this.statusEl.style.display = 'block';
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                this.statusEl.style.display = 'none';
                this.classList.remove('success', 'error');
            }, 5000);
        }
    }

    // ==========================================================================
    // <intent-pay> - Płatność jednym tagiem
    // ==========================================================================
    
    class IntentPay extends HTMLElement {
        static get observedAttributes() {
            return ['amount', 'currency', 'product', 'provider', 'email-field'];
        }

        connectedCallback() {
            this.setupButton();
        }

        setupButton() {
            // Style as button
            this.style.cssText = `
                display: inline-block;
                padding: 12px 24px;
                background: ${this.getProviderColor()};
                color: white;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                text-align: center;
            `;

            this.addEventListener('click', () => this.handlePayment());
        }

        getProviderColor() {
            const colors = {
                paypal: '#0070ba',
                stripe: '#635bff',
                przelewy24: '#d62027'
            };
            return colors[this.getAttribute('provider')] || '#333';
        }

        async handlePayment() {
            const emailField = this.getAttribute('email-field');
            let email = this.getAttribute('email');
            
            // Get email from linked field
            if (emailField) {
                const field = document.querySelector(emailField);
                email = field?.value;
            }
            
            // Prompt for email if not provided
            if (!email) {
                email = prompt('Podaj email do wysłania potwierdzenia:');
                if (!email) return;
            }

            this.textContent = '⏳ Przekierowuję...';
            this.style.opacity = '0.7';

            try {
                const result = await api('payment', {
                    action: 'checkout',
                    amount: parseFloat(this.getAttribute('amount')),
                    currency: this.getAttribute('currency') || 'PLN',
                    product: this.getAttribute('product'),
                    email: email,
                    provider: this.getAttribute('provider') || 'paypal',
                    return_url: this.getAttribute('return-url') || window.location.href,
                    cancel_url: this.getAttribute('cancel-url') || window.location.href
                });

                if (result.redirect_url) {
                    window.location.href = result.redirect_url;
                }

            } catch (error) {
                this.textContent = '❌ Błąd płatności';
                this.style.background = '#dc3545';
                console.error('Payment error:', error);
            }
        }
    }

    // ==========================================================================
    // <intent-image> - Auto-odświeżany obraz
    // ==========================================================================
    
    class IntentImage extends HTMLElement {
        static get observedAttributes() {
            return ['src', 'refresh', 'placeholder'];
        }

        connectedCallback() {
            this.img = document.createElement('img');
            this.img.style.cssText = 'max-width: 100%; height: auto;';
            this.appendChild(this.img);
            
            this.updateImage();
            
            const refresh = parseInt(this.getAttribute('refresh'));
            if (refresh > 0) {
                this.interval = setInterval(() => this.updateImage(), refresh);
            }
        }

        disconnectedCallback() {
            if (this.interval) clearInterval(this.interval);
        }

        updateImage() {
            const src = this.getAttribute('src');
            const separator = src.includes('?') ? '&' : '?';
            this.img.src = `${src}${separator}_t=${Date.now()}`;
        }

        attributeChangedCallback(name, oldVal, newVal) {
            if (name === 'src' && this.img) {
                this.updateImage();
            }
        }
    }

    // ==========================================================================
    // <intent-data> - Dane z bazy danych
    // ==========================================================================
    
    class IntentData extends HTMLElement {
        static get observedAttributes() {
            return ['table', 'limit', 'sort', 'filter', 'template', 'refresh'];
        }

        connectedCallback() {
            this.loadData();
            
            const refresh = parseInt(this.getAttribute('refresh'));
            if (refresh > 0) {
                this.interval = setInterval(() => this.loadData(), refresh);
            }
        }

        disconnectedCallback() {
            if (this.interval) clearInterval(this.interval);
        }

        async loadData() {
            try {
                this.classList.add('loading');
                
                const result = await api('data', {
                    action: 'list',
                    table: this.getAttribute('table'),
                    limit: parseInt(this.getAttribute('limit')) || 10,
                    sort: this.getAttribute('sort'),
                    filter: this.getAttribute('filter')
                });

                this.render(result.items || []);
                this.classList.remove('loading');

            } catch (error) {
                console.error('Data load error:', error);
                this.innerHTML = '<div class="error">Błąd ładowania danych</div>';
            }
        }

        render(items) {
            const templateId = this.getAttribute('template');
            const template = templateId ? document.querySelector(templateId) : null;

            if (template) {
                this.innerHTML = items.map(item => {
                    let html = template.innerHTML;
                    // Simple mustache-style replacement
                    for (const [key, value] of Object.entries(item)) {
                        html = html.replace(new RegExp(`{{${key}}}`, 'g'), value);
                    }
                    return html;
                }).join('');
            } else {
                // Default table rendering
                if (items.length === 0) {
                    this.innerHTML = '<div class="empty">Brak danych</div>';
                    return;
                }

                const headers = Object.keys(items[0]);
                this.innerHTML = `
                    <table class="intent-table">
                        <thead>
                            <tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>
                        </thead>
                        <tbody>
                            ${items.map(item => `
                                <tr>${headers.map(h => `<td>${item[h] ?? ''}</td>`).join('')}</tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
        }
    }

    // ==========================================================================
    // <intent-camera> - Podgląd kamery z AI
    // ==========================================================================
    
    class IntentCamera extends HTMLElement {
        static get observedAttributes() {
            return ['source', 'refresh', 'detect', 'alert-email'];
        }

        connectedCallback() {
            this.innerHTML = `
                <div class="intent-camera-container">
                    <img class="intent-camera-feed" style="width:100%">
                    <div class="intent-camera-overlay"></div>
                    <div class="intent-camera-status"></div>
                </div>
            `;

            this.img = this.querySelector('img');
            this.overlay = this.querySelector('.intent-camera-overlay');
            this.status = this.querySelector('.intent-camera-status');

            this.startStream();
        }

        disconnectedCallback() {
            if (this.interval) clearInterval(this.interval);
        }

        startStream() {
            const refresh = parseInt(this.getAttribute('refresh')) || 1000;
            
            const update = async () => {
                try {
                    const result = await api('camera', {
                        action: 'snapshot',
                        source: this.getAttribute('source')
                    });
                    
                    if (result.image) {
                        this.img.src = `data:image/jpeg;base64,${result.image}`;
                    }

                    // Run detection if enabled
                    const detect = this.getAttribute('detect');
                    if (detect) {
                        this.runDetection(detect.split(','));
                    }

                } catch (error) {
                    this.status.textContent = '❌ Błąd połączenia';
                }
            };

            update();
            this.interval = setInterval(update, refresh);
        }

        async runDetection(types) {
            try {
                const result = await api('camera', {
                    action: 'analyze',
                    source: this.getAttribute('source'),
                    detect: types
                });

                // Draw detection boxes
                this.overlay.innerHTML = (result.detections || []).map(d => `
                    <div class="detection-box" style="
                        position: absolute;
                        left: ${d.bbox.x * 100}%;
                        top: ${d.bbox.y * 100}%;
                        width: ${d.bbox.width * 100}%;
                        height: ${d.bbox.height * 100}%;
                        border: 2px solid ${d.type === 'person' ? 'red' : 'yellow'};
                    ">
                        <span class="detection-label">${d.type} ${(d.confidence * 100).toFixed(0)}%</span>
                    </div>
                `).join('');

                // Send alert if person detected
                const alertEmail = this.getAttribute('alert-email');
                if (alertEmail && result.detections?.some(d => d.type === 'person')) {
                    this.sendAlert(alertEmail, result);
                }

            } catch (error) {
                console.error('Detection error:', error);
            }
        }

        async sendAlert(email, detection) {
            // Throttle alerts (max 1 per minute)
            if (this._lastAlert && Date.now() - this._lastAlert < 60000) return;
            this._lastAlert = Date.now();

            await api('email', {
                action: 'send',
                to: email,
                template: 'camera_alert',
                data: detection
            });
        }
    }

    // ==========================================================================
    // <intent-auth> - Logowanie/Rejestracja
    // ==========================================================================
    
    class IntentAuth extends HTMLElement {
        static get observedAttributes() {
            return ['mode', 'redirect', 'providers'];
        }

        connectedCallback() {
            const mode = this.getAttribute('mode') || 'login';
            const providers = (this.getAttribute('providers') || '').split(',').filter(Boolean);

            this.innerHTML = `
                <div class="intent-auth">
                    <form class="intent-auth-form">
                        ${mode === 'register' ? `
                            <input type="text" name="name" placeholder="Imię i nazwisko" required>
                        ` : ''}
                        <input type="email" name="email" placeholder="Email" required>
                        <input type="password" name="password" placeholder="Hasło" required>
                        ${mode === 'register' ? `
                            <input type="password" name="password_confirm" placeholder="Powtórz hasło" required>
                        ` : ''}
                        <button type="submit">${mode === 'login' ? 'Zaloguj' : 'Zarejestruj'}</button>
                    </form>
                    ${providers.length ? `
                        <div class="intent-auth-divider">lub</div>
                        <div class="intent-auth-social">
                            ${providers.map(p => `
                                <button class="intent-auth-${p}" data-provider="${p}">
                                    ${p === 'google' ? '🔵 Google' : p === 'github' ? '⚫ GitHub' : p}
                                </button>
                            `).join('')}
                        </div>
                    ` : ''}
                    <div class="intent-auth-status"></div>
                </div>
            `;

            this.querySelector('form').addEventListener('submit', (e) => this.handleSubmit(e));
            this.querySelectorAll('[data-provider]').forEach(btn => {
                btn.addEventListener('click', () => this.handleSocial(btn.dataset.provider));
            });
        }

        async handleSubmit(e) {
            e.preventDefault();
            const mode = this.getAttribute('mode') || 'login';
            const formData = Object.fromEntries(new FormData(e.target));
            const status = this.querySelector('.intent-auth-status');

            try {
                const result = await api('auth', {
                    action: mode,
                    ...formData
                });

                if (result.token) {
                    localStorage.setItem('auth_token', result.token);
                    status.textContent = '✅ Sukces!';
                    
                    const redirect = this.getAttribute('redirect');
                    if (redirect) {
                        window.location.href = redirect;
                    }
                }

            } catch (error) {
                status.textContent = '❌ ' + (error.message || 'Błąd logowania');
            }
        }

        async handleSocial(provider) {
            const result = await api('auth', {
                action: 'social',
                provider: provider,
                redirect_url: window.location.href
            });

            if (result.redirect_url) {
                window.location.href = result.redirect_url;
            }
        }
    }

    // ==========================================================================
    // <intent-chart> - Wykres z auto-odświeżaniem
    // ==========================================================================
    
    class IntentChart extends HTMLElement {
        static get observedAttributes() {
            return ['type', 'data-source', 'refresh', 'labels', 'datasets'];
        }

        async connectedCallback() {
            // Load Chart.js if not present
            if (!window.Chart) {
                await this.loadChartJS();
            }

            this.canvas = document.createElement('canvas');
            this.appendChild(this.canvas);

            await this.loadData();

            const refresh = parseInt(this.getAttribute('refresh'));
            if (refresh > 0) {
                this.interval = setInterval(() => this.loadData(), refresh);
            }
        }

        disconnectedCallback() {
            if (this.interval) clearInterval(this.interval);
            if (this.chart) this.chart.destroy();
        }

        async loadChartJS() {
            return new Promise((resolve) => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
                script.onload = resolve;
                document.head.appendChild(script);
            });
        }

        async loadData() {
            const dataSource = this.getAttribute('data-source');
            
            let data;
            if (dataSource) {
                const result = await api('data', {
                    action: 'query',
                    endpoint: dataSource
                });
                data = result;
            } else {
                // Parse inline data
                data = {
                    labels: JSON.parse(this.getAttribute('labels') || '[]'),
                    datasets: JSON.parse(this.getAttribute('datasets') || '[]')
                };
            }

            this.renderChart(data);
        }

        renderChart(data) {
            if (this.chart) {
                this.chart.data = data;
                this.chart.update();
            } else {
                this.chart = new Chart(this.canvas, {
                    type: this.getAttribute('type') || 'line',
                    data: data,
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
            }
        }
    }

    // ==========================================================================
    // Register Components
    // ==========================================================================
    
    customElements.define('intent-form', IntentForm);
    customElements.define('intent-pay', IntentPay);
    customElements.define('intent-image', IntentImage);
    customElements.define('intent-data', IntentData);
    customElements.define('intent-camera', IntentCamera);
    customElements.define('intent-auth', IntentAuth);
    customElements.define('intent-chart', IntentChart);

    // ==========================================================================
    // Default Styles
    // ==========================================================================
    
    const styles = document.createElement('style');
    styles.textContent = `
        intent-form, intent-data, intent-camera, intent-auth {
            display: block;
        }
        
        intent-form.loading button[type="submit"] {
            opacity: 0.6;
            cursor: wait;
        }
        
        .intent-status {
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
        }
        
        .intent-status-success {
            background: #d4edda;
            color: #155724;
        }
        
        .intent-status-error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .intent-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .intent-table th, .intent-table td {
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
        }
        
        .intent-table th {
            background: #f5f5f5;
        }
        
        .intent-camera-container {
            position: relative;
        }
        
        .intent-camera-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
        }
        
        .detection-label {
            position: absolute;
            top: -20px;
            left: 0;
            background: red;
            color: white;
            padding: 2px 6px;
            font-size: 11px;
        }
        
        .intent-auth-form {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .intent-auth-form input {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        .intent-auth-divider {
            text-align: center;
            margin: 15px 0;
            color: #999;
        }
        
        .intent-auth-social {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
    `;
    document.head.appendChild(styles);

    // Export
    window.IntentForgeComponents = {
        IntentForm,
        IntentPay,
        IntentImage,
        IntentData,
        IntentCamera,
        IntentAuth,
        IntentChart
    };

})();
