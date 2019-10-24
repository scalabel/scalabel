import mockfs from 'mock-fs'
import { BundleFile, HandlerUrl,
  ItemTypeName, LabelTypeName } from '../../js/common/types'
import Session from '../../js/server/server_session'
import * as util from '../../js/server/util'
import { sampleFormEmpty, sampleFormImage } from '../test_creation_objects'

beforeAll(() => {
  // mock the file system for testing storage
  mockfs({
    'data/': {}
  })

  // init global env to default
  const defaultEnv = {}
  Session.setEnv(defaultEnv)
  // init global storage
  util.initStorage(Session.getEnv())
})

describe('test general utility methods', () => {
  test('make empty creation form', () => {
    const form = util.makeCreationForm()
    expect(form).toEqual(sampleFormEmpty)
  })

  test('make image creation form', () => {
    const form = util.makeCreationForm(
      'sampleName', ItemTypeName.IMAGE, LabelTypeName.BOX_2D,
      'sampleTitle', 5, 'instructions.com', false
    )
    expect(form).toEqual(sampleFormImage)
  })

  test('file keys', () => {
    const projectName = 'testProject'
    const projectKey = util.getProjectKey(projectName)
    const taskKey = util.getTaskKey(projectName, '000000')
    const savedKey = util.getSavedKey(projectName, '000000')

    expect(projectKey).toBe('testProject/project')
    expect(taskKey).toBe('testProject/tasks/000000')
    expect(savedKey).toBe('testProject/saved/000000')
  })

  test('index2str', () => {
    expect(util.index2str(100)).toBe('000100')
    expect(util.index2str(5)).toBe('000005')
    expect(util.index2str(789631)).toBe('789631')
  })

  test('handler url selection', () => {
    // img item => label2d handler
    let handler = util.getHandlerUrl(
      ItemTypeName.IMAGE, LabelTypeName.BOX_3D)
    expect(handler).toBe(HandlerUrl.LABEL_2D)

    // video item + box2d label => 2d handler
    handler = util.getHandlerUrl(
      ItemTypeName.VIDEO, LabelTypeName.BOX_2D)
    expect(handler).toBe(HandlerUrl.LABEL_2D)

    // video item + other label => invalid handler
    handler = util.getHandlerUrl(
      ItemTypeName.VIDEO, LabelTypeName.BOX_3D)
    expect(handler).toBe(HandlerUrl.INVALID)

    // point cloud item + box3d label => 3d handler
    handler = util.getHandlerUrl(
      ItemTypeName.POINT_CLOUD, LabelTypeName.BOX_3D)
    expect(handler).toBe(HandlerUrl.LABEL_3D)
    handler = util.getHandlerUrl(
      ItemTypeName.POINT_CLOUD_TRACKING, LabelTypeName.BOX_3D)
    expect(handler).toBe(HandlerUrl.LABEL_3D)

    // point cloud item + other label => invalid handler
    handler = util.getHandlerUrl(
      ItemTypeName.POINT_CLOUD, LabelTypeName.BOX_2D)
    expect(handler).toBe(HandlerUrl.INVALID)
    handler = util.getHandlerUrl(
      ItemTypeName.POINT_CLOUD_TRACKING, LabelTypeName.TAG)
    expect(handler).toBe(HandlerUrl.INVALID)

    // any other combinations are invalid
    handler = util.getHandlerUrl(
      'invalid item', LabelTypeName.BOX_2D)
    expect(handler).toBe(HandlerUrl.INVALID)
  })

  test('bundle file selection', () => {
    // tag label => v2 bundle
    let bundleFile = util.getBundleFile(LabelTypeName.TAG)
    expect(bundleFile).toBe(BundleFile.V2)

    // box2d label => v2 bundle
    bundleFile = util.getBundleFile(LabelTypeName.BOX_2D)
    expect(bundleFile).toBe(BundleFile.V2)

    // any other label => v1 bundle
    bundleFile = util.getBundleFile(LabelTypeName.BOX_3D)
    expect(bundleFile).toBe(BundleFile.V1)
    bundleFile = util.getBundleFile(LabelTypeName.POLYGON_2D)
    expect(bundleFile).toBe(BundleFile.V1)
    bundleFile = util.getBundleFile('invalid label')
    expect(bundleFile).toBe(BundleFile.V1)
  })

  test('tracking selection', () => {
    // video => true
    const [itemType1, tracking1] = util.getTracking(ItemTypeName.VIDEO)
    expect(itemType1).toBe(ItemTypeName.IMAGE)
    expect(tracking1).toBe(true)

    // point cloud => true
    const [itemType2, tracking2] = util.getTracking(
      ItemTypeName.POINT_CLOUD_TRACKING)
    expect(itemType2).toBe(ItemTypeName.POINT_CLOUD)
    expect(tracking2).toBe(true)

    // any other item => false
    const [itemType3, tracking3] = util.getTracking(ItemTypeName.IMAGE)
    expect(itemType3).toBe(ItemTypeName.IMAGE)
    expect(tracking3).toBe(false)

    const [itemType4, tracking4] = util.getTracking(ItemTypeName.POINT_CLOUD)
    expect(itemType4).toBe(ItemTypeName.POINT_CLOUD)
    expect(tracking4).toBe(false)
  })
})

// TODO- test utility methods that use storage
// describe('test utility methods that use storage', () => {
// })

afterAll(() => {
  mockfs.restore()
})
