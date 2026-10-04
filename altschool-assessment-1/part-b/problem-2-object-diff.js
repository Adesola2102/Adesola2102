// Problem 2 — Object Diff
// Returns { added, removed, changed } comparing only top-level keys.
function diffObjects(oldObj, newObj) {
  const result = { added: {}, removed: {}, changed: {} }

  const oldKeys = new Set(Object.keys(oldObj))
  const newKeys = new Set(Object.keys(newObj))

  for (const key of newKeys) {
    if (!oldKeys.has(key)) {
      // Key only exists in the new object
      result.added[key] = newObj[key]
    } else if (oldObj[key] !== newObj[key]) {
      // Key exists in both, but the value changed
      result.changed[key] = { from: oldObj[key], to: newObj[key] }
    }
  }

  for (const key of oldKeys) {
    if (!newKeys.has(key)) {
      // Key only exists in the old object
      result.removed[key] = oldObj[key]
    }
  }

  return result
}

console.log(diffObjects(
  { name: 'Setemi', role: 'Engineer', country: 'Jamaica' },
  { name: 'Setemi', role: 'Senior Engineer', city: 'Kingston' }
))
// { added: { city: 'Kingston' }, removed: { country: 'Jamaica' }, changed: { role: { from: 'Engineer', to: 'Senior Engineer' } } }
