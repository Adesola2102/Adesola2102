// Problem 1 — Deep Equal
// Returns true if two values have the same keys and values recursively,
// without using JSON.stringify.
function deepEqual(objA, objB) {
  // Same primitive value or same reference
  if (objA === objB) return true

  // If either is not an object (or is null), they can't be deeply equal here
  if (
    typeof objA !== 'object' || objA === null ||
    typeof objB !== 'object' || objB === null
  ) {
    return false
  }

  // An array should not equal a plain object with the same keys
  if (Array.isArray(objA) !== Array.isArray(objB)) return false

  const keysA = Object.keys(objA)
  const keysB = Object.keys(objB)

  if (keysA.length !== keysB.length) return false

  for (const key of keysA) {
    if (!Object.hasOwn(objB, key)) return false
    if (!deepEqual(objA[key], objB[key])) return false
  }

  return true
}

console.log(deepEqual({ a: 1, b: { c: 2 } }, { a: 1, b: { c: 2 } })) // true
console.log(deepEqual({ a: 1, b: { c: 2 } }, { a: 1, b: { c: 3 } })) // false
console.log(deepEqual({ a: 1 }, { a: 1, b: 2 }))                     // false
