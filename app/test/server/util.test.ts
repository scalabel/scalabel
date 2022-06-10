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
    const handlers = new Map([
      // Image
      [
        ItemTypeName.IMAGE,
        new Map([
          [LabelTypeName.EMPTY, HandlerUrl.LABEL],
          [LabelTypeName.TAG, HandlerUrl.LABEL],
          [LabelTypeName.BOX_2D, HandlerUrl.LABEL],
          [LabelTypeName.POLYGON_2D, HandlerUrl.LABEL],
          [LabelTypeName.POLYLINE_2D, HandlerUrl.LABEL],
          [LabelTypeName.CUSTOM_2D, HandlerUrl.LABEL],
          [LabelTypeName.BOX_3D, HandlerUrl.LABEL],
          [LabelTypeName.PLANE_3D, HandlerUrl.LABEL]
        ])
      ],

      // Video
      [
        ItemTypeName.VIDEO,
        new Map([
          [LabelTypeName.BOX_2D, HandlerUrl.LABEL],
          [LabelTypeName.POLYGON_2D, HandlerUrl.LABEL],
          [LabelTypeName.POLYLINE_2D, HandlerUrl.LABEL],
          [LabelTypeName.CUSTOM_2D, HandlerUrl.LABEL],
          [LabelTypeName.BOX_3D, HandlerUrl.LABEL]
        ])
      ],

      // Point cloud
      [
        ItemTypeName.POINT_CLOUD,
        new Map([[LabelTypeName.BOX_3D, HandlerUrl.LABEL]])
      ],

      // Point cloud tracking
      [
        ItemTypeName.POINT_CLOUD_TRACKING,
        new Map([[LabelTypeName.BOX_3D, HandlerUrl.LABEL]])
      ],

      // Fusion
      [ItemTypeName.FUSION, new Map([[LabelTypeName.BOX_3D, HandlerUrl.LABEL]])]
    ])

    handlers.forEach((labels, item) => {
      labels.forEach((want, label) => {
        const handler = util.getHandlerUrl(item, label)
        expect(handler, `expect ${item} + ${label} => ${want}`).toBe(want)
      })
    })

    // Any other combinations are invalid
    Object.values(ItemTypeName).forEach((item) => {
      Object.values(LabelTypeName).forEach((label) => {
        if (handlers.get(item)?.has(label) ?? false) {
          return
        }
        const handler = util.getHandlerUrl(item, label)
        expect(handler, `expect ${item} + ${label} => invalid`).toBe(
          HandlerUrl.INVALID
        )
      })
    })
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
