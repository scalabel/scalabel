import "@testing-library/jest-dom"

import { PathPoint2DType, SimplePathPoint2DType } from "../../src/types/state"

declare global {
  // TODO: figure out how to extend jest without lint error
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace jest {
    interface Matchers<R> {
      /** Match to two sets of polygon points */
      toMatchPoints2D: (a: Array<Partial<SimplePathPoint2DType>>) => R
    }
  }
}

expect.extend({
  /**
   * Add matching points to jest expect
   *
   * @param received
   * @param argument
   */
  toMatchPoints2D(
    received: PathPoint2DType[],
    argument: Array<Partial<SimplePathPoint2DType>>
  ) {
    if (received.length !== argument.length) {
      return {
        message: () =>
          `expected ${this.utils.printReceived(
            received
          )} to have equal length with ${this.utils.printExpected(argument)}`,
        pass: true
      }
    }
    const num = received.length
    const unequalIndices: number[] = []
    for (let i = 0; i < num; i += 1) {
      const p0 = received[i]
      const p1 = argument[i]
      if (
        (p1.x !== undefined && p0.x !== p1.x) ||
        (p1.y !== undefined && p0.y !== p1.y) ||
        (p1.pointType !== undefined && p0.pointType !== p1.pointType)
      ) {
        unequalIndices.push(i)
      }
    }
    if (unequalIndices.length === 0) {
      return {
        message: () =>
          `expected ${this.utils.printReceived(
            received
          )} not to match points ${this.utils.printExpected(argument)}`,
        pass: true
      }
    } else {
      return {
        message: () =>
          `expected ${this.utils.printReceived(
            received
          )} to match points ${this.utils.printExpected(
            argument
          )} at [${unequalIndices.map((i) => i.toString()).join(", ")}]`,
        pass: false
      }
    }
  }
})
