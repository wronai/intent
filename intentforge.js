/**
 * IntentForge.js - Minimalistyczne SDK dla frontendu
 * 
 * Użycie:
 *   <script src="https://cdn.intentforge.io/v1/intentforge.min.js"></script>
 *   <script>
 *     const api = IntentForge.connect('ws://localhost:9001');
 *     api.form('contact').submit(data);
 *     api.payment.checkout({amount: 29.99, product: 'ebook'});
 *   </script>
 */

(function(global) {
    'use strict';

    const VERSION = '1.0.0';
    const DEFAULT_BROKER = 'ws://localhost:9001';
    const DEFAULT_API = 'http://localhost:8000';
    const TOPIC_PREFIX = 'intentforge';

    // ==========================================================================
    // Core Client
    // ==========================================================================
    
    class IntentForgeClient {
        constructor(options = {}) {
            this.broker = options.broker || DEFAULT_BROKER;
            this.api = options.api || DEFAULT_API;
            this.clientId = options.clientId || `if-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
            this.debug = options.debug || false;
            
            this.mqtt = null;
            this.connected = false;
            this.pending = new Map();
            this.subscriptions = new Map();
            this.eventHandlers = new Map();
            
            // Lazy-load MQTT
            this._mqttPromise = null;
        }

        // ----------------------------------------------------------------------
        // Connection Management
        // ----------------------------------------------------------------------
        
        async connect() {
            if (this.connected) return this;
            
            await this._loadMQTT();
            
            return new Promise((resolve, reject) => {
                const url = new URL(this.broker);
                
                this.mqtt = new Paho.MQTT.Client(
                    url.hostname,
                    parseInt(url.port) || 9001,
                    this.clientId
                );
                
                this.mqtt.onConnectionLost = (res) => {
                    this.connected = false;
                    this._emit('disconnect', res);
                    if (res.errorCode !== 0) {
                        setTimeout(() => this.connect(), 3000);
                    }
                };
                
                this.mqtt.onMessageArrived = (msg) => this._handleMessage(msg);
                
                this.mqtt.connect({
                    onSuccess: () => {
                        this.connected = true;
                        this._subscribe(`${TOPIC_PREFIX}/+/response/${this.clientId}`);
                        this._subscribe(`${TOPIC_PREFIX}/events/${this.clientId}/#`);
                        this._emit('connect');
                        resolve(this);
                    },
                    onFailure: reject,
                    useSSL: url.protocol === 'wss:'
                });
            });
        }

        async _loadMQTT() {
            if (typeof Paho !== 'undefined') return;
            if (this._mqttPromise) return this._mqttPromise;
            
            this._mqttPromise = new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js';
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
            
            return this._mqttPromise;
        }

        // ----------------------------------------------------------------------
        // Form Handling - Najprostsze API
        // ----------------------------------------------------------------------
        
        /**
         * Obsługa formularza - automatyczne generowanie backendu
         * 
         * @example
         * api.form('contact').submit({name: 'Jan', email: 'jan@example.com'});
         * api.form('newsletter').onSuccess(data => console.log('Subscribed!'));
         */
        form(formId) {
            return new FormHandler(this, formId);
        }

        /**
         * Auto-bind wszystkich formularzy na stronie
         * 
         * @example
         * api.autoBindForms(); // Wszystkie <form data-intent="...">
         */
        autoBindForms() {
            document.querySelectorAll('form[data-intent]').forEach(form => {
                const handler = this.form(form.id || form.dataset.intent);
                
                form.addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const data = Object.fromEntries(new FormData(form));
                    
                    try {
                        form.classList.add('if-loading');
                        const result = await handler.submit(data);
                        form.classList.remove('if-loading');
                        form.classList.add('if-success');
                        form.dispatchEvent(new CustomEvent('if:success', { detail: result }));
                        
                        if (form.dataset.intentReset !== 'false') {
                            form.reset();
                        }
                    } catch (error) {
                        form.classList.remove('if-loading');
                        form.classList.add('if-error');
                        form.dispatchEvent(new CustomEvent('if:error', { detail: error }));
                    }
                });
            });
            
            return this;
        }

        // ----------------------------------------------------------------------
        // Payment Integration
        // ----------------------------------------------------------------------
        
        /**
         * Płatności - PayPal, Stripe, etc.
         * 
         * @example
         * api.payment.checkout({
         *   amount: 29.99,
         *   currency: 'PLN',
         *   product: 'ebook-python',
         *   email: 'customer@example.com'
         * });
         */
        get payment() {
            return new PaymentHandler(this);
        }

        // ----------------------------------------------------------------------
        // Email
        // ----------------------------------------------------------------------
        
        /**
         * Wysyłanie emaili
         * 
         * @example
         * api.email.send({
         *   to: 'customer@example.com',
         *   template: 'welcome',
         *   data: {name: 'Jan'}
         * });
         */
        get email() {
            return new EmailHandler(this);
        }

        // ----------------------------------------------------------------------
        // Storage / Database
        // ----------------------------------------------------------------------
        
        /**
         * Operacje na danych
         * 
         * @example
         * api.data('products').list({limit: 10});
         * api.data('orders').create({...});
         * api.data('users').get(123);
         */
        data(table) {
            return new DataHandler(this, table);
        }

        // ----------------------------------------------------------------------
        // Real-time Events
        // ----------------------------------------------------------------------
        
        /**
         * Subskrypcja zdarzeń w czasie rzeczywistym
         * 
         * @example
         * api.on('camera:motion', data => alert('Motion detected!'));
         * api.on('order:new', data => updateDashboard(data));
         */
        on(event, callback) {
            if (!this.eventHandlers.has(event)) {
                this.eventHandlers.set(event, []);
                this._subscribe(`${TOPIC_PREFIX}/events/${this.clientId}/${event}`);
            }
            this.eventHandlers.get(event).push(callback);
            return this;
        }

        off(event, callback) {
            if (this.eventHandlers.has(event)) {
                const handlers = this.eventHandlers.get(event);
                const idx = handlers.indexOf(callback);
                if (idx > -1) handlers.splice(idx, 1);
            }
            return this;
        }

        // ----------------------------------------------------------------------
        // Image / Camera
        // ----------------------------------------------------------------------
        
        /**
         * Operacje na obrazach i kamerach
         * 
         * @example
         * api.camera('rtsp://...').analyze();
         * api.camera('front-door').onMotion(data => notify(data));
         * api.image.refresh('#camera-feed', 5000); // Co 5 sekund
         */
        camera(source) {
            return new CameraHandler(this, source);
        }

        get image() {
            return new ImageHandler(this);
        }

        // ----------------------------------------------------------------------
        // Code Generation
        // ----------------------------------------------------------------------
        
        /**
         * Generowanie kodu
         * 
         * @example
         * const code = await api.generate('Create REST API for products');
         */
        async generate(description, options = {}) {
            return this._request('generate', {
                description,
                intent_type: options.type || 'api_endpoint',
                target_platform: options.platform || 'python_fastapi',
                context: options.context || {}
            });
        }

        // ----------------------------------------------------------------------
        // Internal Methods
        // ----------------------------------------------------------------------
        
        async _request(action, data, timeout = 30000) {
            // Try REST first, fallback to MQTT
            try {
                const response = await fetch(`${this.api}/api/${action}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                if (response.ok) {
                    return response.json();
                }
            } catch (e) {
                this._log('REST failed, trying MQTT', e);
            }
            
            // MQTT fallback
            if (!this.connected) await this.connect();
            
            return new Promise((resolve, reject) => {
                const requestId = this._uuid();
                
                this.pending.set(requestId, {
                    resolve,
                    reject,
                    timeout: setTimeout(() => {
                        this.pending.delete(requestId);
                        reject(new Error('Request timeout'));
                    }, timeout)
                });
                
                this._publish(`${TOPIC_PREFIX}/${action}/request/${this.clientId}`, {
                    request_id: requestId,
                    ...data
                });
            });
        }

        _publish(topic, payload) {
            if (!this.connected) {
                throw new Error('Not connected');
            }
            
            const message = new Paho.MQTT.Message(JSON.stringify(payload));
            message.destinationName = topic;
            message.qos = 1;
            this.mqtt.send(message);
            
            this._log('Published', topic, payload);
        }

        _subscribe(topic) {
            if (this.connected && !this.subscriptions.has(topic)) {
                this.mqtt.subscribe(topic);
                this.subscriptions.set(topic, true);
                this._log('Subscribed', topic);
            }
        }

        _handleMessage(msg) {
            try {
                const payload = JSON.parse(msg.payloadString);
                const topic = msg.destinationName;
                
                this._log('Received', topic, payload);
                
                // Handle pending requests
                if (payload.request_id && this.pending.has(payload.request_id)) {
                    const pending = this.pending.get(payload.request_id);
                    clearTimeout(pending.timeout);
                    this.pending.delete(payload.request_id);
                    
                    if (payload.success !== false) {
                        pending.resolve(payload);
                    } else {
                        pending.reject(new Error(payload.error || 'Request failed'));
                    }
                    return;
                }
                
                // Handle events
                const eventMatch = topic.match(/events\/[^/]+\/(.+)$/);
                if (eventMatch) {
                    const event = eventMatch[1];
                    if (this.eventHandlers.has(event)) {
                        this.eventHandlers.get(event).forEach(cb => cb(payload));
                    }
                }
                
            } catch (e) {
                this._log('Error handling message', e);
            }
        }

        _emit(event, data) {
            if (this.eventHandlers.has(event)) {
                this.eventHandlers.get(event).forEach(cb => cb(data));
            }
        }

        _uuid() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
                const r = Math.random() * 16 | 0;
                return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
            });
        }

        _log(...args) {
            if (this.debug) console.log('[IntentForge]', ...args);
        }
    }

    // ==========================================================================
    // Handlers
    // ==========================================================================
    
    class FormHandler {
        constructor(client, formId) {
            this.client = client;
            this.formId = formId;
            this._onSuccess = null;
            this._onError = null;
        }

        async submit(data) {
            try {
                const result = await this.client._request('form', {
                    form_id: this.formId,
                    action: 'submit',
                    data: data
                });
                
                if (this._onSuccess) this._onSuccess(result);
                return result;
            } catch (error) {
                if (this._onError) this._onError(error);
                throw error;
            }
        }

        onSuccess(callback) {
            this._onSuccess = callback;
            return this;
        }

        onError(callback) {
            this._onError = callback;
            return this;
        }
    }

    class PaymentHandler {
        constructor(client) {
            this.client = client;
        }

        async checkout(options) {
            return this.client._request('payment', {
                action: 'checkout',
                amount: options.amount,
                currency: options.currency || 'PLN',
                product: options.product,
                email: options.email,
                return_url: options.returnUrl || window.location.href,
                cancel_url: options.cancelUrl || window.location.href,
                provider: options.provider || 'paypal'
            });
        }

        async verify(paymentId) {
            return this.client._request('payment', {
                action: 'verify',
                payment_id: paymentId
            });
        }

        async refund(paymentId, amount) {
            return this.client._request('payment', {
                action: 'refund',
                payment_id: paymentId,
                amount: amount
            });
        }
    }

    class EmailHandler {
        constructor(client) {
            this.client = client;
        }

        async send(options) {
            return this.client._request('email', {
                action: 'send',
                to: options.to,
                subject: options.subject,
                template: options.template,
                data: options.data || {},
                attachments: options.attachments || []
            });
        }

        async sendTemplate(template, to, data) {
            return this.send({ template, to, data });
        }
    }

    class DataHandler {
        constructor(client, table) {
            this.client = client;
            this.table = table;
        }

        async list(options = {}) {
            return this.client._request('data', {
                action: 'list',
                table: this.table,
                ...options
            });
        }

        async get(id) {
            return this.client._request('data', {
                action: 'get',
                table: this.table,
                id: id
            });
        }

        async create(data) {
            return this.client._request('data', {
                action: 'create',
                table: this.table,
                data: data
            });
        }

        async update(id, data) {
            return this.client._request('data', {
                action: 'update',
                table: this.table,
                id: id,
                data: data
            });
        }

        async delete(id) {
            return this.client._request('data', {
                action: 'delete',
                table: this.table,
                id: id
            });
        }

        async query(sql, params = {}) {
            return this.client._request('data', {
                action: 'query',
                table: this.table,
                sql: sql,
                params: params
            });
        }
    }

    class CameraHandler {
        constructor(client, source) {
            this.client = client;
            this.source = source;
        }

        async analyze(options = {}) {
            return this.client._request('camera', {
                action: 'analyze',
                source: this.source,
                detect: options.detect || ['motion', 'objects'],
                notify: options.notify || false
            });
        }

        async snapshot() {
            return this.client._request('camera', {
                action: 'snapshot',
                source: this.source
            });
        }

        onMotion(callback) {
            this.client.on(`camera:${this.source}:motion`, callback);
            return this;
        }

        onObject(objectType, callback) {
            this.client.on(`camera:${this.source}:object:${objectType}`, callback);
            return this;
        }

        startStream(elementId, options = {}) {
            const element = document.getElementById(elementId);
            if (!element) return this;
            
            const refresh = options.interval || 1000;
            
            const update = async () => {
                try {
                    const result = await this.snapshot();
                    if (result.image) {
                        element.src = `data:image/jpeg;base64,${result.image}`;
                    }
                } catch (e) {
                    console.error('Stream error', e);
                }
            };
            
            update();
            this._streamInterval = setInterval(update, refresh);
            
            return this;
        }

        stopStream() {
            if (this._streamInterval) {
                clearInterval(this._streamInterval);
            }
            return this;
        }
    }

    class ImageHandler {
        constructor(client) {
            this.client = client;
            this._intervals = new Map();
        }

        /**
         * Auto-refresh obrazu na stronie
         * 
         * @param {string} selector - CSS selector
         * @param {number} interval - Interwał w ms
         * @param {string|function} source - URL lub funkcja zwracająca URL
         */
        refresh(selector, interval, source) {
            const elements = document.querySelectorAll(selector);
            
            elements.forEach(el => {
                const getUrl = typeof source === 'function' 
                    ? source 
                    : () => source || el.dataset.src || el.src;
                
                const update = () => {
                    const url = getUrl();
                    const separator = url.includes('?') ? '&' : '?';
                    el.src = `${url}${separator}_t=${Date.now()}`;
                };
                
                update();
                const id = setInterval(update, interval);
                this._intervals.set(el, id);
            });
            
            return this;
        }

        stopRefresh(selector) {
            const elements = document.querySelectorAll(selector);
            elements.forEach(el => {
                if (this._intervals.has(el)) {
                    clearInterval(this._intervals.get(el));
                    this._intervals.delete(el);
                }
            });
            return this;
        }

        async analyze(imageData, options = {}) {
            return this.client._request('image', {
                action: 'analyze',
                image: imageData,
                detect: options.detect || ['objects', 'text', 'faces']
            });
        }
    }

    // ==========================================================================
    // Static Factory
    // ==========================================================================
    
    const IntentForge = {
        VERSION,
        
        /**
         * Utwórz i połącz klienta
         * 
         * @example
         * const api = await IntentForge.connect('ws://localhost:9001');
         */
        async connect(broker, options = {}) {
            const client = new IntentForgeClient({ broker, ...options });
            await client.connect();
            return client;
        },
        
        /**
         * Utwórz klienta bez automatycznego połączenia
         */
        create(options = {}) {
            return new IntentForgeClient(options);
        },
        
        /**
         * Auto-inicjalizacja z data-attributes
         * 
         * <script src="intentforge.js" 
         *         data-broker="ws://localhost:9001"
         *         data-auto-bind="true">
         */
        async autoInit() {
            const script = document.currentScript || 
                document.querySelector('script[src*="intentforge"]');
            
            if (!script) return null;
            
            const broker = script.dataset.broker || DEFAULT_BROKER;
            const autoBind = script.dataset.autoBind !== 'false';
            const debug = script.dataset.debug === 'true';
            
            const client = await this.connect(broker, { debug });
            
            if (autoBind) {
                client.autoBindForms();
            }
            
            // Expose globally
            global.intentForge = client;
            
            return client;
        }
    };

    // ==========================================================================
    // Export
    // ==========================================================================
    
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = IntentForge;
    } else {
        global.IntentForge = IntentForge;
    }

    // Auto-init on DOM ready
    if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => IntentForge.autoInit());
        } else {
            IntentForge.autoInit();
        }
    }

})(typeof window !== 'undefined' ? window : global);
