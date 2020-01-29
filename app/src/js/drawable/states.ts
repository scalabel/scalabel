
import { addTrack, changeLabelsProps } from '../action/common'
import { ADD_LABELS, CHANGE_SHAPES } from '../action/types'
import Session from '../common/session'
import { Track } from '../common/track/track'
import { LabelType, ShapeType } from '../functional/types'
import Label2D from './2d/label2d'
import Label3D from './3d/label3d'

/**
 * Commit labels to state
 */
export function commitLabels (
  updatedLabelDrawables: Array<Readonly<Label2D | Label3D>>
) {
  // Get labels & tracks to commit
  const updatedShapes: { [index: number]: { [id: number]: ShapeType}} = {}
  const updatedLabels: { [index: number]: { [id: number]: LabelType}} = {}
  const newTracks: Track[] = []
  const newLabels: Array<Readonly<Label2D | Label3D>> = []
  updatedLabelDrawables.forEach((drawable) => {
    drawable.setManual()
    if (drawable.labelId >= 0) {
      // Existing labels & tracks
      if (Session.tracking) {
        if (drawable.trackId in Session.tracks) {
          const track = Session.tracks[drawable.trackId]
          track.update(
            drawable.label.item,
            drawable
          )
          for (const index of track.updatedIndices) {
            if (!(index in updatedShapes)) {
              updatedShapes[index] = {}
            }
            const shapes = track.getShapes(index)
            for (const shape of shapes) {
              updatedShapes[index][shape.id] = shape.shape
            }

            if (!(index in updatedLabels)) {
              updatedLabels[index] = {}
            }
            const label = track.getLabel(index)
            if (label) {
              updatedLabels[index][label.id] = label
            }
          }
          track.clearUpdatedIndices()
        }
      } else {
        const [ids,,shapes] = drawable.shapeStates()
        if (!(drawable.item in updatedShapes)) {
          updatedShapes[drawable.item] = {}
        }
        for (let i = 0; i < ids.length; i++) {
          updatedShapes[drawable.item][ids[i]] = shapes[i]
        }

        if (!(drawable.item in updatedLabels)) {
          updatedLabels[drawable.item] = {}
        }
        updatedLabels[drawable.item][drawable.labelId] =
          drawable.label
      }
    } else {
      // New labels and tracks
      if (Session.tracking) {
        const track = new Track()
        if (track) {
          let parentTrack
          if (
            drawable.parent &&
            drawable.parent.trackId in Session.tracks
          ) {
            parentTrack = Session.tracks[drawable.parent.trackId]
          }
          track.init(
            drawable.item,
            drawable,
            Session.numItems - drawable.item + 1,
            parentTrack
          )
          newTracks.push(track)
        }
      } else {
        newLabels.push(drawable)
      }
    }
  })

  if (Session.tracking && newTracks.length > 0) {
    // Add new tracks to state
    for (const track of newTracks) {
      const indices = []
      const labels = []
      const types = []
      const shapes = []
      for (let i = 0; i < Session.numItems; i++) {
        const label = track.getLabel(i)
        const currentTypes = []
        const currentShapes = []
        if (label) {
          const indexedShapes = track.getShapes(i)
          for (const indexedShape of indexedShapes) {
            currentTypes.push(indexedShape.type)
            currentShapes.push(indexedShape.shape)
          }
          indices.push(i)
          labels.push(label)
          types.push(currentTypes)
          shapes.push(currentShapes)
        }
      }
      Session.dispatch(addTrack(
        indices, track.type, labels, types, shapes
      ))
    }
  } else if (!Session.tracking && newLabels.length > 0) {
    // Add new labels to state
    const labels = []
    const types = []
    const shapes = []
    for (const label of newLabels) {
      labels.push(label.label)
      const [, shapeTypes, shapeStates] = label.shapeStates()
      types.push(shapeTypes)
      shapes.push(shapeStates)
    }
    Session.dispatch(
      {
        type: ADD_LABELS,
        sessionId: Session.id,
        itemIndices: [newLabels[0].item],
        labels: [labels],
        shapeTypes: [types],
        shapes: [shapes]
      }
    )
  }

  if (Object.keys(updatedShapes).length > 0) {
    // Update existing shapes
    const indices = Object.keys(updatedShapes).map((index) => Number(index))
    const ids = []
    const shapes = []
    for (const index of indices) {
      const indexIds = []
      const indexShapes = []
      for (const key of Object.keys(updatedShapes[index])) {
        const shapeId = Number(key)
        indexIds.push(shapeId)
        indexShapes.push(updatedShapes[index][shapeId])
      }
      ids.push(indexIds)
      shapes.push(indexShapes)
    }
    Session.dispatch(
      {
        type: CHANGE_SHAPES,
        sessionId: Session.id,
        itemIndices: indices,
        shapeIds: ids,
        shapes
      }
    )
  }

  if (Object.keys(updatedLabels).length > 0) {
    // Update existing labels
    const indices = Object.keys(updatedLabels).map((index) => Number(index))
    const ids = []
    const labels = []
    for (const index of indices) {
      const indexIds = []
      const indexLabels = []
      for (const key of Object.keys(updatedLabels[index])) {
        const shapeId = Number(key)
        indexIds.push(shapeId)
        indexLabels.push(updatedLabels[index][shapeId])
      }
      ids.push(indexIds)
      labels.push(indexLabels)
    }
    Session.dispatch(changeLabelsProps(indices, ids, labels))
  }
  Session.label3dList.clearUpdatedLabels()
}
