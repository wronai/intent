/**
 * IntentForge MQTT Client
 * Enables NLP-driven code generation from any static HTML page
 *
 * Usage:
 * <script src="intentforge-client.js"></script>
 * <script>
 *   const forge = new IntentForgeMQTT({broker: 'ws://localhost:9001'});
 *   forge.sendIntent("Create API endpoint for users");
 * </script>
 */

class IntentForgeMQTT {
    constructor(options = {}) {
        this.broker = options.broker || 'ws://localhost:9001';
        this.clientId = options.clientId || `intentforge-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        this.reconnectInterval = options.reconnectInterval || 5000;
        this.topicPrefix = options.topicPrefix || 'intentforge';

        this.client = null;
        this.connected = false;
        this.subscriptions = new Map();
        this.pendingRequests = new Map();

        // Event handlers
        this.onConnect = options.onConnect || (() => {});
        this.onDisconnect = options.onDisconnect || (() => {});
        this.onError = options.onError || console.error;
        this.onMessage = options.onMessage || (() => {});

        // Auto-connect if specified
        if (options.autoConnect !== false) {
            this.connect();
        }
    }

    /**
     * Connect to MQTT broker
     */
    connect() {
        return new Promise((resolve, reject) => {
            // Use Paho MQTT client
            if (typeof Paho === 'undefined') {
                // Load Paho library dynamically
                const script = document.createElement('script');
                script.src = 'https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js';
                script.onload = () => this._initClient(resolve, reject);
                script.onerror = () => reject(new Error('Failed to load MQTT library'));
                document.head.appendChild(script);
            } else {
                this._initClient(resolve, reject);
            }
        });
    }

    _initClient(resolve, reject) {
        const url = new URL(this.broker);

        this.client = new Paho.MQTT.Client(
            url.hostname,
            parseInt(url.port) || 9001,
            this.clientId
        );

        this.client.onConnectionLost = (response) => {
            this.connected = false;
            this.onDisconnect(response);

            // Auto-reconnect
            setTimeout(() => this.connect(), this.reconnectInterval);
        };

        this.client.onMessageArrived = (message) => {
            this._handleMessage(message);
        };

        this.client.connect({
            onSuccess: () => {
                this.connected = true;

                // Subscribe to response topics
                this.subscribe(`${this.topicPrefix}/intent/response/${this.clientId}`);
                this.subscribe(`${this.topicPrefix}/intent/status/${this.clientId}`);

                this.onConnect();
                resolve();
            },
            onFailure: (err) => {
                this.onError(err);
                reject(err);
            },
            useSSL: url.protocol === 'wss:'
        });
    }

    /**
     * Disconnect from broker
     */
    disconnect() {
        if (this.client && this.connected) {
            this.client.disconnect();
            this.connected = false;
        }
    }

    /**
     * Subscribe to a topic
     */
    subscribe(topic, callback) {
        if (!this.connected) {
            console.warn('Not connected, queueing subscription');
            return;
        }

        this.client.subscribe(topic);

        if (callback) {
            this.subscriptions.set(topic, callback);
        }
    }

    /**
     * Publish message to topic
     */
    publish(topic, payload, qos = 1) {
        if (!this.connected) {
            throw new Error('Not connected to broker');
        }

        const message = new Paho.MQTT.Message(
            typeof payload === 'string' ? payload : JSON.stringify(payload)
        );
        message.destinationName = topic;
        message.qos = qos;

        this.client.send(message);
    }

    /**
     * Send intent for code generation
     * Returns promise with generated code
     */
    sendIntent(description, options = {}) {
        return new Promise((resolve, reject) => {
            const requestId = crypto.randomUUID();

            const intent = {
                request_id: requestId,
                description: description,
                intent_type: options.intentType || 'api_endpoint',
                target_platform: options.targetPlatform || 'python_fastapi',
                context: options.context || {},
                constraints: options.constraints || []
            };

            // Store pending request
            this.pendingRequests.set(requestId, {
                resolve,
                reject,
                timeout: setTimeout(() => {
                    this.pendingRequests.delete(requestId);
                    reject(new Error('Intent request timeout'));
                }, options.timeout || 60000)
            });

            // Publish intent
            this.publish(
                `${this.topicPrefix}/intent/request/${this.clientId}`,
                intent
            );
        });
    }

    /**
     * Generate code for form submission
     */
    generateFormHandler(formId, options = {}) {
        const form = document.getElementById(formId);
        if (!form) {
            throw new Error(`Form with id "${formId}" not found`);
        }

        // Extract form structure
        const fields = Array.from(form.elements)
            .filter(el => el.name && el.type !== 'submit')
            .map(el => ({
                name: el.name,
                type: el.type || 'text',
                required: el.required
            }));

        return this.sendIntent(
            `Create API endpoint to handle form submission with fields: ${fields.map(f => f.name).join(', ')}`,
            {
                intentType: 'api_endpoint',
                context: {
                    form_id: formId,
                    fields: fields,
                    method: form.method || 'POST',
                    action: options.apiEndpoint || `/api/${formId}`
                },
                ...options
            }
        );
    }

    /**
     * Generate database query from natural language
     */
    generateQuery(description, options = {}) {
        return this.sendIntent(description, {
            intentType: 'database_schema',
            context: {
                database_type: options.databaseType || 'postgresql',
                use_orm: options.useOrm !== false
            },
            ...options
        });
    }

    /**
     * Handle incoming messages
     */
    _handleMessage(message) {
        try {
            const payload = JSON.parse(message.payloadString);
            const topic = message.destinationName;

            // Check for pending requests
            if (payload.request_id && this.pendingRequests.has(payload.request_id)) {
                const pending = this.pendingRequests.get(payload.request_id);
                clearTimeout(pending.timeout);
                this.pendingRequests.delete(payload.request_id);

                if (payload.success) {
                    pending.resolve(payload);
                } else {
                    pending.reject(new Error(payload.error || 'Request failed'));
                }
                return;
            }

            // Check for subscription callbacks
            if (this.subscriptions.has(topic)) {
                this.subscriptions.get(topic)(payload);
            }

            // General message handler
            this.onMessage(topic, payload);

        } catch (e) {
            this.onError(e);
        }
    }
}


/**
 * DOM Event Observer for automatic intent generation
 * Watches for specific events and generates code based on natural language attributes
 */
class IntentObserver {
    constructor(mqttClient) {
        this.mqtt = mqttClient;
        this.observers = [];
    }

    /**
     * Start observing elements with data-intent attribute
     */
    observe(rootElement = document.body) {
        // Find all elements with intent declarations
        const elements = rootElement.querySelectorAll('[data-intent]');

        elements.forEach(el => {
            const intent = el.dataset.intent;
            const event = el.dataset.intentEvent || 'click';
            const target = el.dataset.intentTarget;

            el.addEventListener(event, async (e) => {
                e.preventDefault();

                try {
                    el.classList.add('intent-processing');

                    const result = await this.mqtt.sendIntent(intent, {
                        context: this._extractContext(el),
                        targetPlatform: el.dataset.intentPlatform
                    });

                    // Handle result
                    if (target) {
                        document.querySelector(target).innerHTML =
                            `<pre><code>${this._escapeHtml(result.generated_code)}</code></pre>`;
                    }

                    el.dispatchEvent(new CustomEvent('intentComplete', { detail: result }));

                } catch (error) {
                    el.dispatchEvent(new CustomEvent('intentError', { detail: error }));
                } finally {
                    el.classList.remove('intent-processing');
                }
            });
        });
    }

    /**
     * Extract context from element and its form
     */
    _extractContext(el) {
        const context = {};

        // Get data attributes
        for (const [key, value] of Object.entries(el.dataset)) {
            if (key.startsWith('intentContext')) {
                const contextKey = key.replace('intentContext', '').toLowerCase();
                context[contextKey] = value;
            }
        }

        // If inside a form, get form data
        const form = el.closest('form');
        if (form) {
            const formData = new FormData(form);
            context.formData = Object.fromEntries(formData.entries());
        }

        return context;
    }

    _escapeHtml(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
}


// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { IntentForgeMQTT, IntentObserver };
}

// Auto-initialize on DOM ready
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        // Auto-create global instance if config exists
        const config = document.querySelector('script[data-intentforge-config]');
        if (config) {
            try {
                const options = JSON.parse(config.dataset.intentforgeConfig);
                window.intentForge = new IntentForgeMQTT(options);
                window.intentObserver = new IntentObserver(window.intentForge);
                window.intentObserver.observe();
            } catch (e) {
                console.error('Failed to initialize IntentForge:', e);
            }
        }
    });
}
