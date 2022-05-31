import { getTracking } from "../../src/common/util"
import {
  BundleFile,
  HandlerUrl,
  ItemTypeName,
  LabelTypeName
} from "../../src/const/common"
import { getProjectKey, getSaveDir, getTaskKey } from "../../src/server/path"
import * as util from "../../src/server/util"
import {
  sampleFormEmpty,
  sampleFormImage
} from "../test_states/test_creation_objects"

describe("test general utility methods", () => {
  test("make empty creation form", () => {
    const form = util.makeCreationForm()
    expect(form).toEqual(sampleFormEmpty)
  })

  test("make image creation form", () => {
    const form = util.makeCreationForm(
      "sampleName",
      ItemTypeName.IMAGE,
      LabelTypeName.BOX_2D,
      "sampleTitle",
      5,
      1,
      "instructions.com",
      false
    )
    expect(form).toEqual(sampleFormImage)
  })

  test("file keys", () => {
    const projectName = "testProject"
    const projectKey = getProjectKey(projectName)
    const taskKey = getTaskKey(projectName, "000000")
    const saveDir = getSaveDir(projectName, "000000")

    expect(projectKey).toBe("projects/testProject/project")
    expect(taskKey).toBe("projects/testProject/tasks/000000")
    expect(saveDir).toBe("projects/testProject/saved/000000")
  })

  test("Init session id", () => {
    expect(util.initSessionId("sampleId")).toBe("sampleId")
    expect(util.initSessionId("")).not.toBe("")
  })

  test("handler url selection", () => {
    const testcases = new Map<[ItemTypeName, LabelTypeName], HandlerUrl>([
      // Image
      [[ItemTypeName.IMAGE, LabelTypeName.EMPTY], HandlerUrl.LABEL],
      [[ItemTypeName.IMAGE, LabelTypeName.TAG], HandlerUrl.LABEL],
      [[ItemTypeName.IMAGE, LabelTypeName.BOX_2D], HandlerUrl.LABEL],
      [[ItemTypeName.IMAGE, LabelTypeName.POLYGON_2D], HandlerUrl.LABEL],
      [[ItemTypeName.IMAGE, LabelTypeName.POLYLINE_2D], HandlerUrl.LABEL],
      [[ItemTypeName.IMAGE, LabelTypeName.CUSTOM_2D], HandlerUrl.LABEL],
      [[ItemTypeName.IMAGE, LabelTypeName.BOX_3D], HandlerUrl.LABEL],
      [[ItemTypeName.IMAGE, LabelTypeName.PLANE_3D], HandlerUrl.LABEL],

      // Video
      [[ItemTypeName.VIDEO, LabelTypeName.BOX_2D], HandlerUrl.LABEL],
      [[ItemTypeName.VIDEO, LabelTypeName.POLYGON_2D], HandlerUrl.LABEL],
      [[ItemTypeName.VIDEO, LabelTypeName.POLYLINE_2D], HandlerUrl.LABEL],
      [[ItemTypeName.VIDEO, LabelTypeName.CUSTOM_2D], HandlerUrl.LABEL],
      [[ItemTypeName.VIDEO, LabelTypeName.BOX_3D], HandlerUrl.LABEL],

      // Point cloud
      [[ItemTypeName.POINT_CLOUD, LabelTypeName.BOX_3D], HandlerUrl.LABEL],

      // Point cloud tracking
      [
        [ItemTypeName.POINT_CLOUD_TRACKING, LabelTypeName.BOX_3D],
        HandlerUrl.LABEL
      ],

      // Fusion
      [[ItemTypeName.FUSION, LabelTypeName.BOX_3D], HandlerUrl.LABEL]
    ])

    testcases.forEach((want, [item, label]) => {
      const handler = util.getHandlerUrl(item, label)
      expect(handler).toBe(want)
    })

    // Any other combinations are invalid
    for (const item in ItemTypeName) {
      for (const label in LabelTypeName) {
        if (testcases.has([item as ItemTypeName, label as LabelTypeName])) {
          continue
        }
        const handler = util.getHandlerUrl(item, label)
        expect(handler).toBe(HandlerUrl.INVALID)
      }
    }
  })

  test("bundle file selection", () => {
    // Tag label => v2 bundle
    let bundleFile = util.getBundleFile(LabelTypeName.TAG)
    expect(bundleFile).toBe(BundleFile.V2)

    // Box2d label => v2 bundle
    bundleFile = util.getBundleFile(LabelTypeName.BOX_2D)
    expect(bundleFile).toBe(BundleFile.V2)

    // Any other label => v1 bundle
    bundleFile = util.getBundleFile(LabelTypeName.BOX_3D)
    expect(bundleFile).toBe(BundleFile.V1)
    bundleFile = util.getBundleFile(LabelTypeName.POLYGON_2D)
    expect(bundleFile).toBe(BundleFile.V1)
    bundleFile = util.getBundleFile("invalid label")
    expect(bundleFile).toBe(BundleFile.V1)
  })

  test("tracking selection", () => {
    // Video => true
    const [itemType1, tracking1] = getTracking(ItemTypeName.VIDEO)
    expect(itemType1).toBe(ItemTypeName.IMAGE)
    expect(tracking1).toBe(true)

    // Point cloud => true
    const [itemType2, tracking2] = getTracking(
      ItemTypeName.POINT_CLOUD_TRACKING
    )
    expect(itemType2).toBe(ItemTypeName.POINT_CLOUD)
    expect(tracking2).toBe(true)

    // Any other item => false
    const [itemType3, tracking3] = getTracking(ItemTypeName.IMAGE)
    expect(itemType3).toBe(ItemTypeName.IMAGE)
    expect(tracking3).toBe(false)

    const [itemType4, tracking4] = getTracking(ItemTypeName.POINT_CLOUD)
    expect(itemType4).toBe(ItemTypeName.POINT_CLOUD)
    expect(tracking4).toBe(false)
  })
})
