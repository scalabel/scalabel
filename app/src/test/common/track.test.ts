import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initFromJson } from '../../js/common/session_init'
import { makeTrackPolicy, Track } from '../../js/common/track'
import { Box2D } from '../../js/drawable/2d/box2d'
import { Rect2D } from '../../js/drawable/2d/rect2d'
import { Box3D } from '../../js/drawable/3d/box3d'
import { Cube3D } from '../../js/drawable/3d/cube3d'
import { makeTrack } from '../../js/functional/states'
import { CubeType, RectType } from '../../js/functional/types'
import { Size2D } from '../../js/math/size2d'
import { Vector2D } from '../../js/math/vector2d'
import { Vector3D } from '../../js/math/vector3d'
import { testJson as imgTestJson } from '../test_image_objects'
import { testJson as pcTestJson } from '../test_point_cloud_objects'
import { expectRectTypesClose, expectVector3TypesClose } from '../util'

test('box3d linear interpolation tracking', () => {
  Session.devMode = false
  pcTestJson.task.config.tracking = true
  initFromJson(pcTestJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(action.changeSelect({ policyType: 0 }))

  let state = Session.getState()
  const currentPolicyType =
    state.task.config.policyTypes[state.user.select.policyType]
  const newTrack = new Track()
  newTrack.updateState(
    makeTrack(-1), makeTrackPolicy(newTrack, currentPolicyType)
  )
  Session.tracks[-1] = newTrack

  const box = new Box3D()
  box.init(state)

  state = Session.getState()

  let trackId = -1
  let firstLabel
  let lastLabel
  for (let i = 0; i < state.task.items.length; i++) {
    const item = state.task.items[i]
    const labels = item.labels
    const keys = Object.keys(labels)
    expect(keys.length).toEqual(1)

    const labelId = Number(keys[0])
    if (trackId < 0) {
      trackId = labels[labelId].track
    } else {
      expect(labels[labelId].track).toEqual(trackId)
    }

    if (i === 0) {
      firstLabel = labels[labelId]
    } else if (i === state.task.items.length - 1) {
      lastLabel = labels[labelId]
    }

    // Set all to be not manual
    Session.dispatch(action.changeLabelProps(
      labels[labelId].item, labels[labelId].id, { manual: false }
    ))
  }

  expect(trackId in Session.tracks).toEqual(true)

  expect(firstLabel).not.toBeNull()
  expect(lastLabel).not.toBeNull()
  expect(firstLabel).not.toBeUndefined()
  expect(lastLabel).not.toBeUndefined()

  if (firstLabel && lastLabel) {
    const newProps: CubeType = (box.shapes()[0] as Cube3D).toCube()
    newProps.center = { x: 5, y: 5, z: 5 }
    Session.dispatch(action.changeLabelShape(
      firstLabel.item,
      firstLabel.shapes[0],
      newProps
    ))
    Session.dispatch(action.changeLabelProps(
      firstLabel.item, firstLabel.id, { manual: true }
    ))

    if (Session.tracking && trackId in Session.tracks) {
      Session.tracks[trackId].onLabelUpdated(
        firstLabel.item, [newProps]
      )
    }

    state = Session.getState()
    for (const item of state.task.items) {
      const labels = item.labels
      const keys = Object.keys(labels)
      expect(keys.length).toEqual(1)

      if (item.index > 0) {
        const label = labels[Number(keys[0])]
        const shape = item.shapes[label.shapes[0]].shape as CubeType
        expectVector3TypesClose(shape.center, (new Vector3D()).toObject())
        expect(label.manual).toEqual(false)
      }
    }

    newProps.center = { x: 0, y: 0, z: 0 }
    Session.dispatch(action.changeLabelShape(
      lastLabel.item,
      lastLabel.shapes[0],
      newProps
    ))
    Session.dispatch(action.changeLabelProps(
      lastLabel.item, lastLabel.id, { manual: true }
    ))

    if (Session.tracking && trackId in Session.tracks) {
      Session.tracks[trackId].onLabelUpdated(
        lastLabel.item, [newProps]
      )
    }

    state = Session.getState()
    let currentDelta
    for (let i = 0; i < state.task.items.length - 1; i++) {
      const currentItem = state.task.items[i]
      const currentLabels = currentItem.labels
      const currentLabelId = Number(Object.keys(currentLabels)[0])
      const nextItem = state.task.items[i + 1]
      const nextLabels = nextItem.labels
      const nextLabelId = Number(Object.keys(nextLabels)[0])

      const currentLabel = currentLabels[currentLabelId]
      const currentShape =
        currentItem.shapes[currentLabel.shapes[0]].shape as CubeType
      const nextLabel = nextLabels[nextLabelId]
      const nextShape =
        nextItem.shapes[nextLabel.shapes[0]].shape as CubeType

      const currentCenter = (new Vector3D()).fromObject(currentShape.center)
      const nextCenter = (new Vector3D()).fromObject(nextShape.center)
      const newDelta = currentCenter.subtract(nextCenter)
      if (currentDelta) {
        expectVector3TypesClose(currentDelta, newDelta)
      }
      currentDelta = newDelta
    }
  }
})

test('box2d linear interpolation tracking', () => {
  Session.devMode = false
  imgTestJson.task.config.tracking = true
  initFromJson(imgTestJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(action.changeSelect({ policyType: 0 }))

  let state = Session.getState()
  const currentPolicyType =
    state.task.config.policyTypes[state.user.select.policyType]
  const newTrack = new Track()
  newTrack.updateState(
    makeTrack(-1), makeTrackPolicy(newTrack, currentPolicyType)
  )
  Session.tracks[-1] = newTrack

  const box = new Box2D()
  box.initTemp(state, new Vector2D())
  box.onMouseDown(new Vector2D())
  box.onMouseMove(new Vector2D(5, 5), new Size2D(50, 50), 0, 0)
  box.onMouseUp(new Vector2D(5, 5))

  state = Session.getState()

  let trackId = -1
  let firstLabel
  let lastLabel
  for (let i = 0; i < state.task.items.length; i++) {
    const item = state.task.items[i]
    const labels = item.labels
    const keys = Object.keys(labels)
    expect(keys.length).toEqual(1)

    const labelId = Number(keys[0])
    if (trackId < 0) {
      trackId = labels[labelId].track
    } else {
      expect(labels[labelId].track).toEqual(trackId)
    }

    if (i === 0) {
      firstLabel = labels[labelId]
    } else if (i === state.task.items.length - 1) {
      lastLabel = labels[labelId]
    }

    // Set all to be not manual
    Session.dispatch(action.changeLabelProps(
      labels[labelId].item, labels[labelId].id, { manual: false }
    ))
  }

  expect(trackId in Session.tracks).toEqual(true)

  expect(firstLabel).not.toBeNull()
  expect(lastLabel).not.toBeNull()
  expect(firstLabel).not.toBeUndefined()
  expect(lastLabel).not.toBeUndefined()

  const originalProps: RectType = (box.shapes[0] as Rect2D).toRect()
  if (firstLabel && lastLabel) {
    let newProps: RectType = { x1: -2.5, y1: 2.5, x2: 2.5, y2: -2.5 }
    Session.dispatch(action.changeLabelShape(
      firstLabel.item,
      firstLabel.shapes[0],
      newProps
    ))
    Session.dispatch(action.changeLabelProps(
      firstLabel.item, firstLabel.id, { manual: true }
    ))

    if (Session.tracking && trackId in Session.tracks) {
      Session.tracks[trackId].onLabelUpdated(
        firstLabel.item, [newProps]
      )
    }

    state = Session.getState()
    for (const item of state.task.items) {
      const labels = item.labels
      const keys = Object.keys(labels)
      expect(keys.length).toEqual(1)

      if (item.index > 0) {
        const label = labels[Number(keys[0])]
        const shape = item.shapes[label.shapes[0]].shape as RectType
        expectRectTypesClose(originalProps, shape)
        expect(label.manual).toEqual(false)
      }
    }

    newProps = { x1: 0, y1: 5, x2: 5, y2: 0 }
    Session.dispatch(action.changeLabelShape(
      lastLabel.item,
      lastLabel.shapes[0],
      newProps
    ))
    Session.dispatch(action.changeLabelProps(
      lastLabel.item, lastLabel.id, { manual: true }
    ))

    if (Session.tracking && trackId in Session.tracks) {
      Session.tracks[trackId].onLabelUpdated(
        lastLabel.item, [newProps]
      )
    }

    state = Session.getState()
    let currentDelta
    for (let i = 0; i < state.task.items.length - 1; i++) {
      const currentItem = state.task.items[i]
      const currentLabels = currentItem.labels
      const currentLabelId = Number(Object.keys(currentLabels)[0])
      const nextItem = state.task.items[i + 1]
      const nextLabels = nextItem.labels
      const nextLabelId = Number(Object.keys(nextLabels)[0])

      const currentLabel = currentLabels[currentLabelId]
      const currentShape =
        currentItem.shapes[currentLabel.shapes[0]].shape as RectType
      const nextLabel = nextLabels[nextLabelId]
      const nextShape =
        nextItem.shapes[nextLabel.shapes[0]].shape as RectType

      const currentCenter = new Vector2D(
        (currentShape.x1 + currentShape.x2) / 2.,
        (currentShape.y1 + currentShape.y2) / 2.
      )
      const nextCenter = new Vector2D(
        (nextShape.x1 + nextShape.x2) / 2.,
        (nextShape.y1 + nextShape.y2) / 2.
      )
      const newDelta = currentCenter.subtract(nextCenter)
      if (currentDelta) {
        expect(currentDelta[0]).toBeCloseTo(newDelta[0], 2)
        expect(currentDelta[1]).toBeCloseTo(newDelta[1], 2)
      }
      currentDelta = newDelta
    }
  }
})
