/**
 * IntentForge SDK v2.0
 * Unified, Secure Frontend SDK
 * 
 * Features:
 * - REST + MQTT dual mode
 * - Built-in security (rate limiting, sanitization, CSRF)
 * - Offline support
 * - TypeScript-ready
 * 
 * Usage:
 *   <script src="intentforge.min.js"></script>
 *   <script>
 *     const api = await IntentForge.init({ mode: 'rest' });
 *     api.form('contact').submit(data);
 *   </script>
 */

(function(global) {
    'use strict';

    const VERSION = '2.0.0';
    
    // ==========================================================================
    // Configuration
    // ==========================================================================
    
    const DEFAULT_CONFIG = {
        // Connection
        apiUrl: 'http://localhost:8000',
        mqttBroker: 'ws://localhost:9001',
        mode: 'auto',  // 'rest', 'mqtt', 'auto' (try REST first)
        
        // Security
        enableRateLimit: true,
        rateLimitPerMinute: 60,
        enableSanitization: true,
        enableCSRF: true,
        csrfTokenName: 'X-CSRF-Token',
        
        // Timeouts
        requestTimeout: 30000,
        connectionTimeout: 10000,
        
        // Retry
        maxRetries: 3,
        retryDelay: 1000,
        
        // Offline
        enableOfflineQueue: true,
        maxQueueSize: 100,
        
        // Debug
        debug: false
    };
    
    // ==========================================================================
    // Security Module
    // ==========================================================================
    
    class Security {
        constructor(config) {
            this.config = config;
            this.requestCounts = new Map();
            this.csrfToken = null;
        }
        
        /**
         * Rate limiting - track requests per minute
         */
        checkRateLimit() {
            if (!this.config.enableRateLimit) return true;
            
            const now = Date.now();
            const minute = Math.floor(now / 60000);
            const key = `rate_${minute}`;
            
            const count = this.requestCounts.get(key) || 0;
            
            if (count >= this.config.rateLimitPerMinute) {
                throw new RateLimitError(
                    `Rate limit exceeded. Max ${this.config.rateLimitPerMinute} requests/minute`
                );
            }
            
            this.requestCounts.set(key, count + 1);
            
            // Cleanup old entries
            for (const [k] of this.requestCounts) {
                if (!k.endsWith(minute.toString())) {
                    this.requestCounts.delete(k);
                }
            }
            
            return true;
        }
        
        /**
         * Sanitize input data
         */
        sanitize(data) {
            if (!this.config.enableSanitization) return data;
            
            if (typeof data === 'string') {
                return this._sanitizeString(data);
            }
            
            if (Array.isArray(data)) {
                return data.map(item => this.sanitize(item));
            }
            
            if (data && typeof data === 'object') {
                const sanitized = {};
                for (const [key, value] of Object.entries(data)) {
                    sanitized[this._sanitizeString(key)] = this.sanitize(value);
                }
                return sanitized;
            }
            
            return data;
        }
        
        _sanitizeString(str) {
            if (typeof str !== 'string') return str;
            
            // Remove script tags
            str = str.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
            
            // Escape HTML entities
            str = str
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
            
            // Remove null bytes
            str = str.replace(/\0/g, '');
            
            return str;
        }
        
        /**
         * Validate form data before sending
         */
        validate(data, rules = {}) {
            const errors = [];
            
            for (const [field, fieldRules] of Object.entries(rules)) {
                const value = data[field];
                
                // Required
                if (fieldRules.required && !value) {
                    errors.push({ field, message: `${field} is required` });
                    continue;
                }
                
                if (!value) continue;
                
                // Email
                if (fieldRules.type === 'email') {
                    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                    if (!emailRegex.test(value)) {
                        errors.push({ field, message: 'Invalid email format' });
                    }
                }
                
                // Min length
                if (fieldRules.minLength && value.length < fieldRules.minLength) {
                    errors.push({ 
                        field, 
                        message: `${field} must be at least ${fieldRules.minLength} characters` 
                    });
                }
                
                // Max length
                if (fieldRules.maxLength && value.length > fieldRules.maxLength) {
                    errors.push({ 
                        field, 
                        message: `${field} must be at most ${fieldRules.maxLength} characters` 
                    });
                }
                
                // Pattern
                if (fieldRules.pattern && !new RegExp(fieldRules.pattern).test(value)) {
                    errors.push({ 
                        field, 
                        message: fieldRules.patternMessage || `${field} has invalid format` 
                    });
                }
            }
            
            return {
                valid: errors.length === 0,
                errors
            };
        }
        
        /**
         * Get CSRF token
         */
        async getCSRFToken() {
            if (!this.config.enableCSRF) return null;
            
            if (this.csrfToken) return this.csrfToken;
            
            // Try to get from cookie
            const cookie = document.cookie
                .split(';')
                .find(c => c.trim().startsWith('csrf_token='));
            
            if (cookie) {
                this.csrfToken = cookie.split('=')[1];
                return this.csrfToken;
            }
            
            // Request new token
            try {
                const response = await fetch(`${this.config.apiUrl}/api/csrf`, {
                    credentials: 'include'
                });
                const data = await response.json();
                this.csrfToken = data.token;
                return this.csrfToken;
            } catch (e) {
                console.warn('Could not get CSRF token:', e);
                return null;
            }
        }
    }
    
    // ==========================================================================
    // Offline Queue
    // ==========================================================================
    
    class OfflineQueue {
        constructor(config) {
            this.config = config;
            this.queue = [];
            this.processing = false;
            
            // Listen for online event
            if (typeof window !== 'undefined') {
                window.addEventListener('online', () => this.processQueue());
            }
        }
        
        add(request) {
            if (!this.config.enableOfflineQueue) return false;
            
            if (this.queue.length >= this.config.maxQueueSize) {
                // Remove oldest
                this.queue.shift();
            }
            
            this.queue.push({
                ...request,
                timestamp: Date.now()
            });
            
            // Store in localStorage
            this._persist();
            
            return true;
        }
        
        async processQueue() {
            if (this.processing || this.queue.length === 0) return;
            if (!navigator.onLine) return;
            
            this.processing = true;
            
            while (this.queue.length > 0) {
                const request = this.queue[0];
                
                try {
                    // Attempt to send
                    await this._sendRequest(request);
                    this.queue.shift();
                    this._persist();
                } catch (e) {
                    // Failed, stop processing
                    break;
                }
            }
            
            this.processing = false;
        }
        
        async _sendRequest(request) {
            // Implementation depends on request type
            // Will be set by main client
        }
        
        _persist() {
            try {
                localStorage.setItem('intentforge_queue', JSON.stringify(this.queue));
            } catch (e) {
                // Ignore storage errors
            }
        }
        
        _restore() {
            try {
                const stored = localStorage.getItem('intentforge_queue');
                if (stored) {
                    this.queue = JSON.parse(stored);
                }
            } catch (e) {
                // Ignore
            }
        }
    }
    
    // ==========================================================================
    // HTTP Client (REST mode)
    // ==========================================================================
    
    class HTTPClient {
        constructor(config, security) {
            this.config = config;
            this.security = security;
        }
        
        async request(endpoint, data = {}, options = {}) {
            // Rate limit check
            this.security.checkRateLimit();
            
            // Sanitize data
            const sanitizedData = this.security.sanitize(data);
            
            // Build headers
            const headers = {
                'Content-Type': 'application/json',
                ...options.headers
            };
            
            // Add CSRF token
            const csrfToken = await this.security.getCSRFToken();
            if (csrfToken) {
                headers[this.config.csrfTokenName] = csrfToken;
            }
            
            // Build URL
            const url = `${this.config.apiUrl}${endpoint}`;
            
            // Retry logic
            let lastError;
            for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
                try {
                    const controller = new AbortController();
                    const timeout = setTimeout(
                        () => controller.abort(),
                        this.config.requestTimeout
                    );
                    
                    const response = await fetch(url, {
                        method: options.method || 'POST',
                        headers,
                        body: JSON.stringify(sanitizedData),
                        credentials: 'include',
                        signal: controller.signal
                    });
                    
                    clearTimeout(timeout);
                    
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({}));
                        throw new APIError(
                            errorData.message || `HTTP ${response.status}`,
                            response.status,
                            errorData
                        );
                    }
                    
                    return await response.json();
                    
                } catch (error) {
                    lastError = error;
                    
                    if (error.name === 'AbortError') {
                        throw new TimeoutError('Request timeout');
                    }
                    
                    if (attempt < this.config.maxRetries) {
                        await this._delay(this.config.retryDelay * (attempt + 1));
                    }
                }
            }
            
            throw lastError;
        }
        
        _delay(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }
    }
    
    // ==========================================================================
    // MQTT Client
    // ==========================================================================
    
    class MQTTClient {
        constructor(config, security) {
            this.config = config;
            this.security = security;
            this.client = null;
            this.connected = false;
            this.pending = new Map();
            this.subscriptions = new Map();
            this._mqttPromise = null;
        }
        
        async connect() {
            if (this.connected) return;
            
            await this._loadMQTT();
            
            return new Promise((resolve, reject) => {
                const url = new URL(this.config.mqttBroker);
                const clientId = `if-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
                
                this.client = new Paho.MQTT.Client(
                    url.hostname,
                    parseInt(url.port) || 9001,
                    clientId
                );
                
                this.client.onConnectionLost = () => {
                    this.connected = false;
                    // Auto-reconnect
                    setTimeout(() => this.connect(), 3000);
                };
                
                this.client.onMessageArrived = (msg) => this._handleMessage(msg);
                
                this.client.connect({
                    onSuccess: () => {
                        this.connected = true;
                        this.client.subscribe(`intentforge/+/response/${clientId}`);
                        resolve();
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
        
        async request(topic, data, timeout = 30000) {
            if (!this.connected) {
                await this.connect();
            }
            
            // Rate limit
            this.security.checkRateLimit();
            
            // Sanitize
            const sanitizedData = this.security.sanitize(data);
            
            const requestId = crypto.randomUUID();
            
            return new Promise((resolve, reject) => {
                this.pending.set(requestId, {
                    resolve,
                    reject,
                    timeout: setTimeout(() => {
                        this.pending.delete(requestId);
                        reject(new TimeoutError('MQTT request timeout'));
                    }, timeout)
                });
                
                const message = new Paho.MQTT.Message(JSON.stringify({
                    request_id: requestId,
                    ...sanitizedData
                }));
                message.destinationName = topic;
                message.qos = 1;
                
                this.client.send(message);
            });
        }
        
        _handleMessage(msg) {
            try {
                const payload = JSON.parse(msg.payloadString);
                
                if (payload.request_id && this.pending.has(payload.request_id)) {
                    const pending = this.pending.get(payload.request_id);
                    clearTimeout(pending.timeout);
                    this.pending.delete(payload.request_id);
                    
                    if (payload.success !== false) {
                        pending.resolve(payload);
                    } else {
                        pending.reject(new APIError(payload.error || 'Request failed'));
                    }
                }
            } catch (e) {
                // Ignore parse errors
            }
        }
    }
    
    // ==========================================================================
    // Main Client
    // ==========================================================================
    
    class IntentForgeClient {
        constructor(config) {
            this.config = { ...DEFAULT_CONFIG, ...config };
            this.security = new Security(this.config);
            this.http = new HTTPClient(this.config, this.security);
            this.mqtt = new MQTTClient(this.config, this.security);
            this.offline = new OfflineQueue(this.config);
            
            this._mode = this.config.mode;
        }
        
        /**
         * Make request (auto-selects transport)
         */
        async request(service, action, data = {}) {
            // Check offline
            if (!navigator.onLine) {
                if (this.config.enableOfflineQueue) {
                    this.offline.add({ service, action, data });
                    return { queued: true };
                }
                throw new OfflineError('No internet connection');
            }
            
            // Select transport
            if (this._mode === 'mqtt') {
                return this.mqtt.request(`intentforge/${service}/request`, {
                    action,
                    ...data
                });
            }
            
            // REST (default)
            return this.http.request(`/api/${service}`, {
                action,
                ...data
            });
        }
        
        // ------------------------------------------------------------------
        // Form Handler
        // ------------------------------------------------------------------
        
        form(formId) {
            return new FormHandler(this, formId);
        }
        
        autoBindForms(options = {}) {
            const selector = options.selector || 'form[data-intent]';
            
            document.querySelectorAll(selector).forEach(form => {
                const intent = form.dataset.intent || form.id;
                const handler = this.form(intent);
                
                form.addEventListener('submit', async (e) => {
                    e.preventDefault();
                    
                    const submitBtn = form.querySelector('[type="submit"]');
                    const originalText = submitBtn?.textContent;
                    
                    try {
                        form.classList.add('if-loading');
                        if (submitBtn) {
                            submitBtn.disabled = true;
                            submitBtn.textContent = options.loadingText || 'Sending...';
                        }
                        
                        const formData = Object.fromEntries(new FormData(form));
                        const result = await handler.submit(formData);
                        
                        form.classList.remove('if-loading');
                        form.classList.add('if-success');
                        form.dispatchEvent(new CustomEvent('if:success', { detail: result }));
                        
                        if (form.dataset.reset !== 'false') {
                            form.reset();
                        }
                        
                    } catch (error) {
                        form.classList.remove('if-loading');
                        form.classList.add('if-error');
                        form.dispatchEvent(new CustomEvent('if:error', { detail: error }));
                    } finally {
                        if (submitBtn) {
                            submitBtn.disabled = false;
                            submitBtn.textContent = originalText;
                        }
                    }
                });
            });
            
            return this;
        }
        
        // ------------------------------------------------------------------
        // Payment
        // ------------------------------------------------------------------
        
        get payment() {
            return {
                checkout: (options) => this.request('payment', 'checkout', options),
                verify: (paymentId) => this.request('payment', 'verify', { payment_id: paymentId })
            };
        }
        
        // ------------------------------------------------------------------
        // Email
        // ------------------------------------------------------------------
        
        get email() {
            return {
                send: (options) => this.request('email', 'send', options)
            };
        }
        
        // ------------------------------------------------------------------
        // Data
        // ------------------------------------------------------------------
        
        data(table) {
            return {
                list: (options) => this.request('data', 'list', { table, ...options }),
                get: (id) => this.request('data', 'get', { table, id }),
                create: (data) => this.request('data', 'create', { table, data }),
                update: (id, data) => this.request('data', 'update', { table, id, data }),
                delete: (id) => this.request('data', 'delete', { table, id })
            };
        }
        
        // ------------------------------------------------------------------
        // Camera
        // ------------------------------------------------------------------
        
        camera(source) {
            return new CameraHandler(this, source);
        }
        
        // ------------------------------------------------------------------
        // Image Auto-refresh
        // ------------------------------------------------------------------
        
        get image() {
            return new ImageHandler();
        }
        
        // ------------------------------------------------------------------
        // Events
        // ------------------------------------------------------------------
        
        on(event, callback) {
            // MQTT events
            if (this._mode === 'mqtt' || this._mode === 'auto') {
                this.mqtt.subscriptions.set(event, callback);
            }
            return this;
        }
    }
    
    // ==========================================================================
    // Handler Classes
    // ==========================================================================
    
    class FormHandler {
        constructor(client, formId) {
            this.client = client;
            this.formId = formId;
            this._rules = {};
            this._onSuccess = null;
            this._onError = null;
        }
        
        rules(rules) {
            this._rules = rules;
            return this;
        }
        
        onSuccess(callback) {
            this._onSuccess = callback;
            return this;
        }
        
        onError(callback) {
            this._onError = callback;
            return this;
        }
        
        async submit(data) {
            // Validate first
            const validation = this.client.security.validate(data, this._rules);
            if (!validation.valid) {
                const error = new ValidationError('Validation failed', validation.errors);
                if (this._onError) this._onError(error);
                throw error;
            }
            
            try {
                const result = await this.client.request('form', 'submit', {
                    form_id: this.formId,
                    data
                });
                
                if (this._onSuccess) this._onSuccess(result);
                return result;
                
            } catch (error) {
                if (this._onError) this._onError(error);
                throw error;
            }
        }
    }
    
    class CameraHandler {
        constructor(client, source) {
            this.client = client;
            this.source = source;
            this._streamInterval = null;
        }
        
        async analyze(options = {}) {
            return this.client.request('camera', 'analyze', {
                source: this.source,
                ...options
            });
        }
        
        async snapshot() {
            return this.client.request('camera', 'snapshot', {
                source: this.source
            });
        }
        
        startStream(elementId, interval = 1000) {
            const element = document.getElementById(elementId);
            if (!element) return this;
            
            const update = async () => {
                try {
                    const result = await this.snapshot();
                    if (result.image) {
                        element.src = `data:image/jpeg;base64,${result.image}`;
                    }
                } catch (e) {
                    console.error('Stream error:', e);
                }
            };
            
            update();
            this._streamInterval = setInterval(update, interval);
            return this;
        }
        
        stopStream() {
            if (this._streamInterval) {
                clearInterval(this._streamInterval);
                this._streamInterval = null;
            }
            return this;
        }
        
        onMotion(callback) {
            this.client.on(`camera:${this.source}:motion`, callback);
            return this;
        }
    }
    
    class ImageHandler {
        constructor() {
            this._intervals = new Map();
        }
        
        refresh(selector, interval, source) {
            document.querySelectorAll(selector).forEach(el => {
                const getUrl = typeof source === 'function'
                    ? source
                    : () => source || el.dataset.src || el.src;
                
                const update = () => {
                    const url = getUrl();
                    const sep = url.includes('?') ? '&' : '?';
                    el.src = `${url}${sep}_t=${Date.now()}`;
                };
                
                update();
                const id = setInterval(update, interval);
                this._intervals.set(el, id);
            });
            
            return this;
        }
        
        stopRefresh(selector) {
            document.querySelectorAll(selector).forEach(el => {
                if (this._intervals.has(el)) {
                    clearInterval(this._intervals.get(el));
                    this._intervals.delete(el);
                }
            });
            return this;
        }
    }
    
    // ==========================================================================
    // Error Classes
    // ==========================================================================
    
    class IntentForgeError extends Error {
        constructor(message) {
            super(message);
            this.name = 'IntentForgeError';
        }
    }
    
    class APIError extends IntentForgeError {
        constructor(message, status, data) {
            super(message);
            this.name = 'APIError';
            this.status = status;
            this.data = data;
        }
    }
    
    class ValidationError extends IntentForgeError {
        constructor(message, errors) {
            super(message);
            this.name = 'ValidationError';
            this.errors = errors;
        }
    }
    
    class RateLimitError extends IntentForgeError {
        constructor(message) {
            super(message);
            this.name = 'RateLimitError';
        }
    }
    
    class TimeoutError extends IntentForgeError {
        constructor(message) {
            super(message);
            this.name = 'TimeoutError';
        }
    }
    
    class OfflineError extends IntentForgeError {
        constructor(message) {
            super(message);
            this.name = 'OfflineError';
        }
    }
    
    // ==========================================================================
    // Factory
    // ==========================================================================
    
    const IntentForge = {
        VERSION,
        
        /**
         * Initialize client
         */
        async init(config = {}) {
            const client = new IntentForgeClient(config);
            
            // Auto-connect MQTT if needed
            if (config.mode === 'mqtt' || config.mode === 'auto') {
                try {
                    await client.mqtt.connect();
                } catch (e) {
                    if (config.debug) console.warn('MQTT connection failed:', e);
                    client._mode = 'rest';
                }
            }
            
            return client;
        },
        
        /**
         * Create without connecting
         */
        create(config = {}) {
            return new IntentForgeClient(config);
        },
        
        /**
         * Auto-init from data attributes
         */
        async autoInit() {
            const script = document.currentScript ||
                document.querySelector('script[src*="intentforge"]');
            
            if (!script) return null;
            
            const config = {
                apiUrl: script.dataset.api || DEFAULT_CONFIG.apiUrl,
                mqttBroker: script.dataset.broker || DEFAULT_CONFIG.mqttBroker,
                mode: script.dataset.mode || 'rest',
                debug: script.dataset.debug === 'true'
            };
            
            const client = await this.init(config);
            
            if (script.dataset.autoBind !== 'false') {
                client.autoBindForms();
            }
            
            global.intentForge = client;
            return client;
        },
        
        // Errors
        IntentForgeError,
        APIError,
        ValidationError,
        RateLimitError,
        TimeoutError,
        OfflineError
    };
    
    // ==========================================================================
    // Export
    // ==========================================================================
    
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = IntentForge;
    } else {
        global.IntentForge = IntentForge;
    }
    
    // Auto-init
    if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => IntentForge.autoInit());
        } else {
            IntentForge.autoInit();
        }
    }

})(typeof window !== 'undefined' ? window : global);
