// Problem 4 — Private Counter Factory
// Returns an object with increment(), decrement() and a `value` getter.
// The count lives in a closure, so it cannot be read or changed directly.
function createCounter() {
  let count = 0

  return {
    increment() {
      count++
    },
    decrement() {
      count--
    },
    get value() {
      return count
    },
  }
}

const counter = createCounter()
counter.increment()
counter.increment()
counter.decrement()
console.log(counter.value) // 1
console.log(counter.count) // undefined — not directly accessible
