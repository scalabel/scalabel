import _ from "lodash"

import { Vector } from "../../src/math/vector"
import { Vector2D } from "../../src/math/vector2d"

test("Test basic vector operations", () => {
  const v1 = new Vector(3)
  v1[0] = 1
  v1[1] = 1.1
  v1[2] = 1.2
  const v2 = v1.clone()
  v1.add(v2)
  expect(v1[0]).toEqual(2)
  expect(v1[1]).toEqual(2.2)
  expect(v2[0]).toEqual(1)
  v1.add(1).scale(2)
  expect(v1[0]).toEqual(6)
  v1.add(v2)
  expect(v1[0]).toEqual(7)
  expect(v1[1]).toEqual(7.5)
  v2[0] = 2
  v1.dot(v2)
  expect(v1[0]).toEqual(14)
  expect(v2[0]).toEqual(2)

  // Absolute values
  const v3 = new Vector(3).fill(0).subtract(v1)
  for (const i of _.range(3)) {
    expect(v1[i]).toEqual(-v3[i])
  }
  v3.abs()
  for (const i of _.range(3)) {
    expect(v1[i]).toEqual(v3[i])
  }

  // Product
  expect(v3.prod()).toEqual(v3[0] * v3[1] * v3[2])
})

test("Test basic Vector2D", () => {
  const v1 = new Vector2D()
  v1.x = 1.2
  v1.y = 1.3
  const v2 = v1.clone()
  v2.y = 1.4
  expect(v1.x).toEqual(1.2)
  expect(v1.y).toEqual(1.3)
  expect(v2.x).toEqual(1.2)
  expect(v2.y).toEqual(1.4)
  v1.add(v2)
  expect(v1.y).toEqual(2.7)
})
