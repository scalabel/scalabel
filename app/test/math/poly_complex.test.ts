import { polyIsComplex } from "../../src/math/poly_complex"

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
  const vertices: number[][] = [[]]
  for (let i = 0; i < 10000; i++) {
    vertices.push([Math.random() * 100, Math.random() * 100])
  }
  const start = performance.now()
  polyIsComplex(vertices)
  const end = performance.now()
  expect(start - end).toBeLessThan(1000)
})
