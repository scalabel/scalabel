import _ from 'lodash'
import { updateLabelsShapesTracks } from '../action/common'
import Session from '../common/session'
import { Track } from '../common/track/track'
import { makeIndexedShape, makeLabel, makeTrack } from '../functional/states'
import { IndexedShapeType, LabelType, TrackType } from '../functional/types'
import Label2D from './2d/label2d'
import { Shape2D } from './2d/shape2d'
import Label3D from './3d/label3d'
import { Shape3D } from './3d/shape3d'

/**
 * Commit labels to state
 */
export function commitLabels (
  updatedLabelDrawables: Array<Readonly<Label2D | Label3D>>,
  updatedShapeDrawables: Array<Readonly<Shape2D | Shape3D>>
) {
  // Get labels, shapes, & tracks to commit
  const itemIndices: Set<number> = new Set()
  const updatedShapes: {
    [index: number]: { [id: number]: IndexedShapeType }
  } = {}
  const updatedLabels: { [index: number]: { [id: number]: LabelType}} = {}
  const newTracks: Array<Readonly<TrackType>> = []
  let minNewLabelId = 0
  let minNewShapeId = 0
  updatedLabelDrawables.forEach((drawable) => {
    drawable.setManual()
    if (!(drawable.item in updatedLabels)) {
      updatedLabels[drawable.item] = {}
    }
    updatedLabels[drawable.item][drawable.labelId] = drawable.labelState
    itemIndices.add(drawable.item)
    minNewLabelId = Math.min(drawable.labelId, minNewLabelId)
  })

  updatedShapeDrawables.forEach((shape) => {
    if (!(shape.item in updatedShapes)) {
      updatedShapes[shape.item] = {}
    }
    updatedShapes[shape.item][shape.shapeId] = shape.toState()
    itemIndices.add(shape.item)
    minNewShapeId = Math.min(shape.shapeId, minNewShapeId)
  })

  if (Session.tracking) {
    updatedLabelDrawables.forEach((drawable) => {
      let track
      let newTrackState
      if (drawable.labelId >= 0) {
        // Existing labels & tracks
        if (drawable.trackId in Session.tracks) {
          track = Session.tracks[drawable.trackId]
          track.update(
            drawable.labelState.item,
            drawable
          )
        }
      } else {
        // Remove from updated array, will be replaced by track
        for (const shapeId of drawable.labelState.shapes) {
          delete updatedShapes[drawable.item][shapeId]
        }
        delete updatedLabels[drawable.item][drawable.labelId]
        // New labels and tracks
        track = new Track()
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
          Session.numItems - drawable.item,
          parentTrack
        )
        newTrackState = makeTrack(
          track.state.id,
          track.state.type,
          { ...track.state.labels }
        )
        newTracks.push(newTrackState)
        itemIndices.add(drawable.item)
      }
      if (track) {
        for (const index of track.updatedIndices) {
          if (!(index in updatedShapes)) {
            updatedShapes[index] = {}
          }
          const shapes = track.getShapes(index)
          const shapeIds = []
          for (const shape of shapes) {
            const newShape = makeIndexedShape(
              shape.id, shape.item, shape.labels, shape.type, { ...shape.shape }
            )
            if (shape.id < 0) {
              newShape.id = minNewShapeId
              minNewShapeId--
            }
            updatedShapes[index][newShape.id] = newShape
            shapeIds.push(newShape.id)
          }

          if (!(index in updatedLabels)) {
            updatedLabels[index] = {}
          }
          const label = track.getLabel(index)
          if (label) {
            const newLabel = makeLabel(label)
            newLabel.shapes = shapeIds
            if (label.id < 0) {
              newLabel.id = minNewLabelId

              if (newTrackState && index in newTrackState.labels) {
                newTrackState.labels[index] = minNewLabelId
              }
              minNewLabelId--
            }
            updatedLabels[index][newLabel.id] = newLabel
          }
          itemIndices.add(index)
        }
        track.clearUpdatedIndices()
      }
    })
  }

  const allLabels = []
  const allShapes = []
  for (const index of itemIndices) {
    if (index in updatedLabels) {
      allLabels.push(Object.values(updatedLabels[index]))
    } else {
      allLabels.push([])
    }
    if (index in updatedShapes) {
      allShapes.push(Object.values(updatedShapes[index]))
    } else {
      allShapes.push([])
    }
  }

  Session.dispatch(updateLabelsShapesTracks(
    Array.from(itemIndices), allLabels, allShapes, newTracks
  ))
}
