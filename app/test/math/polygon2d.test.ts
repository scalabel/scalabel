import { mergeNearbyVertices, polyIsComplex } from "../../src/math/polygon2d"

test("Test for vertices merging", () => {
  const testPolygon1: Array<[number, number]> = [
    [1.0, 1.0],
    [1.0, 2.0],
    [2.0, 1.0]
  ]
  const testPolygon2: Array<[number, number]> = [
    [1.0, 1.0],
    [1.1, 1.1],
    [1.0, 2.0],
    [2.0, 1.0]
  ]
  expect(mergeNearbyVertices(testPolygon1, 1)).toMatchObject(testPolygon1)
  expect(mergeNearbyVertices(testPolygon2, 1)).toMatchObject(testPolygon1)
})

test("Test for complex polygons correctness", () => {
  expect(
    polyIsComplex([
      [10, 10],
      [20, 10],
      [20, 20],
      [10, 20],
      [10, 10]
    ])
  ).toEqual(expect.arrayContaining([]))
  expect(
    polyIsComplex([
      [895.4950561523438, 418.41326904296875],
      [895.4950561523438, 474.0956115722656],
      [981.058837890625, 474.0956115722656],
      [981.058837890625, 418.41326904296875],
      [895.4950561523438, 418.41326904296875]
    ])
  ).toEqual(expect.arrayContaining([]))
  expect(
    polyIsComplex([
      [10, 10],
      [20, 10],
      [10, 20],
      [20, 20],
      [10, 10]
    ])
  ).toEqual(expect.arrayContaining([[20, 10, 10, 20, 20, 20, 10, 10]]))
  expect(
    polyIsComplex([
      [895.4950561523438, 418.41326904296875],
      [981.058837890625, 474.0956115722656],
      [895.4950561523438, 474.0956115722656],
      [981.058837890625, 418.41326904296875],
      [895.4950561523438, 418.41326904296875]
    ])
  ).toEqual(
    expect.arrayContaining([
      [
        895.4950561523438,
        418.41326904296875,
        981.058837890625,
        474.0956115722656,
        895.4950561523438,
        474.0956115722656,
        981.058837890625,
        418.41326904296875
      ]
    ])
  )
})

test("Test for complex polygons speed", () => {
  const vertices: Array<[number, number]> = []
  for (let i = 0; i < 1000; i++) {
    vertices.push([Math.random() * 100, Math.random() * 100])
  }
  const start = performance.now()
  polyIsComplex(vertices)
  const end = performance.now()
  expect(start - end).toBeLessThan(10)
})
