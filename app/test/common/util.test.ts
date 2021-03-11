import * as util from "../../src/common/util"

describe("Test shared utility methods", () => {
  test("Convert index to padded string", () => {
    expect(util.index2str(100)).toBe("000100")
    expect(util.index2str(5)).toBe("000005")
    expect(util.index2str(789631)).toBe("789631")
  })

  test("Get correct instruction links", () => {
    expect(util.getInstructionUrl("box2d")).toBe(
      "https://doc.scalabel.ai/instructions/bbox.html"
    )
    expect(util.getInstructionUrl("polygon2d")).toBe(
      "https://doc.scalabel.ai/instructions/segmentation.html"
    )
  })
})
