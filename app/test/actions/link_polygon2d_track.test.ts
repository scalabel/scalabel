import * as action from "../../src/action/common"
import * as track from "../../src/action/track"
import Session from "../../src/common/session"
import { setupTestStore } from "../components/util"
import { makePolygon2dLabel } from "../../src/action/polygon2d"
import { isValidId, makeSimplePathPoint2D } from "../../src/functional/states"
import {
  State,
  LabelIdMap,
  LabelType,
  PathPointType,
  ShapeType
} from "../../src/types/state"
import { testJson } from "../test_states/test_track_polygon2d_objects"

const getState = Session.getState.bind(Session)
const dispatch = Session.dispatch.bind(Session)

beforeEach(() => {
  setupTestStore(testJson)
  Session.subscribe(() => {
    Session.label2dList.updateState(getState())
  })
})

beforeAll(() => {
  setupTestStore(testJson)
  Session.images.length = 0
  for (let i = 0; i < getState().task.items.length; i++) {
    Session.images.push({ [-1]: new Image(1000, 1000) })
    dispatch(action.loadItem(i, -1))
  }
})

describe("Test link labels", () => {
  test("Link polygons", () => {
    const numFrames = getState().task.items.length

    const itemIdx = 0
    dispatch(action.goToItem(itemIdx))

    // add 8 tracks
    for (let i = 0; i < 8; i++) {
      const o = i * 2
      const points = [
        makeSimplePathPoint2D(0 + o, 1 + o, PathPointType.LINE),
        makeSimplePathPoint2D(1 + o, 1 + o, PathPointType.LINE),
        makeSimplePathPoint2D(1 + o, 0 + o, PathPointType.LINE)
      ]
      const frames: number[] = []
      const labels: LabelType[] = []
      const shapes: ShapeType[][] = []
      for (let j = 0; j < numFrames; j++) {
        const [l, ss] = makePolygon2dLabel(-1, [0], points, true)
        l.track = `track_${i}`
        frames.push(j)
        labels.push(l)
        shapes.push(ss)
      }
      dispatch(action.addTrack(frames, "", labels, shapes))
    }

    const flids: string[][] = []
    for (let j = 0; j < numFrames; j++) {
      const lids = Object.keys(getState().task.items[j].labels)
      flids.push(lids)
    }

    // link label 0, 1, 2 in two steps
    dispatch(action.linkLabels(0, [flids[0][0], flids[0][1]]))
    dispatch(action.linkLabels(0, [flids[0][1], flids[0][2]]))

    // link label 3, 4, 5 in one step
    dispatch(action.linkLabels(0, [flids[0][3], flids[0][4], flids[0][5]]))

    // link label 6, 7 in one step
    dispatch(action.linkLabels(0, [flids[0][6], flids[0][7]]))

    // link them all
    dispatch(action.linkLabels(0, [flids[0][0], flids[0][3], flids[0][6]]))

    // link already-linked labels should do no harm
    dispatch(action.linkLabels(0, [flids[0][0], flids[0][3], flids[0][6]]))

    // check state
    let state = getState()
    expectConsistentState(state)

    // `labels` of `track` must correspond to `track` of `labels`
    Object.entries(state.task.tracks).forEach(([_tid, track]) => {
      Object.entries(track.labels).forEach(([idx, lid]) => {
        const label = state.task.items[idx as unknown as number].labels[lid]
        expect(label.track).toEqual(track.id)
      })
    })

    // should be only one track
    let tids = Object.keys(state.task.tracks)
    expect(tids.length).toBe(1)

    // break track at frame 1
    dispatch(action.splitTrack(tids[0], "track_tail", 1))
    state = getState()
    expectConsistentState(state)

    // should be two tracks now
    tids = Object.keys(state.task.tracks)
    expect(tids.length).toBe(2)

    // re-link tracks
    dispatch(action.mergeTracks(tids))
    state = getState()
    expectConsistentState(state)

    // should be one track again
    tids = Object.keys(state.task.tracks)
    expect(tids.length).toBe(1)

    // the track in earch frame should be of following tree structure
    /*
      root
      - p1
        - q1
          - 0
          - 1
        - 2
      - p2
        - 3
        - 4
        - 5
      - p3
        - 6
        - 7
    */
    const tid = tids[0]
    for (let j = 0; j < numFrames; j++) {
      const labels = state.task.items[j].labels
      const rootId = state.task.tracks[tid].labels[j]

      const [l0, l1, l2, l3, l4, l5, l6, l7] = flids[j]
      const p1 = labels[l2].parent
      const p2 = labels[l3].parent
      const p3 = labels[l6].parent
      const q1 = labels[l0].parent
      expectTree(
        {
          [p1]: {
            [q1]: {
              [l0]: {},
              [l1]: {}
            },
            [l2]: {}
          },
          [p2]: {
            [l3]: {},
            [l4]: {},
            [l5]: {}
          },
          [p3]: {
            [l6]: {},
            [l7]: {}
          }
        },
        rootId,
        labels
      )
    }

    // delete the track
    const tracks = Object.values(state.task.tracks)
    dispatch(track.deleteTracks(tracks))
    state = getState()
    expectConsistentState(state)

    // should be no tracks and no labels
    tids = Object.keys(state.task.tracks)
    expect(tids.length).toBe(0)
    Object.entries(state.task.items).forEach(([_idx, frame]) => {
      expect(Object.keys(frame.labels).length).toBe(0)
    })
  })
})

/**
 * Check the state is consistent.
 *
 * @param state
 */
function expectConsistentState(state: State): void {
  state.task.items.forEach((frame) => {
    Object.entries(frame.labels).forEach(([_id, label]) => {
      if (isValidId(label.parent)) {
        // parent label must exist
        const p = frame.labels[label.parent]
        expect(p).toBeTruthy()
      }

      // check track
      const tid = label.track

      // each label must belongs to a track
      expect(label.track).toBeTruthy()

      // track must exists
      const track = state.task.tracks[tid]
      expect(track).toBeTruthy()
    })
  })
}

interface TreeNode {
  [key: string]: TreeNode
}

/**
 * Check the label tree starting at `rootId` has expected structure.
 *
 * @param rootId
 * @param want
 * @param labels
 */
function expectTree(want: TreeNode, rootId: string, labels: LabelIdMap): void {
  const root = labels[rootId]
  const cids = Object.keys(want)
  expect([...root.children].sort()).toEqual(cids.sort())

  root.children.forEach((cid) => {
    const sub = want[cid]
    expectTree(sub, cid, labels)
  })
}
