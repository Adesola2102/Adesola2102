// Problem 3 — Deep Freeze
// Recursively freezes an object and all of its nested objects, then returns it.
function deepFreeze(obj) {
  for (const value of Object.values(obj)) {
    // Freeze nested objects first (skip null and already-frozen objects)
    if (typeof value === 'object' && value !== null && !Object.isFrozen(value)) {
      deepFreeze(value)
    }
  }
  return Object.freeze(obj)
}

const config = deepFreeze({ api: { baseUrl: 'https://x.com', retries: 3 }, debug: false })
config.api.baseUrl = 'https://changed.com' // should be ignored
config.debug = true                        // should be ignored
console.log(config.api.baseUrl, config.debug) // "https://x.com" false
console.log(Object.isFrozen(config.api))      // true
