import _ from 'lodash'
import { updateLabelsShapesTracks } from '../action/common'
import Session from '../common/session'
import { Track } from '../common/track/track'
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
  updatedLabelDrawables.forEach((drawable) => {
    drawable.setManual()
    if (drawable.labelId >= 0) {
      // Existing labels & tracks
      if (Session.tracking) {
        if (drawable.trackId in Session.tracks) {
          const track = Session.tracks[drawable.trackId]
          track.update(
            drawable.labelState.item,
            drawable
          )
          for (const index of track.updatedIndices) {
            if (!(index in updatedShapes)) {
              updatedShapes[index] = {}
            }
            const shapes = track.getShapes(index)
            for (const shape of shapes) {
              updatedShapes[index][shape.id] = shape
            }

            if (!(index in updatedLabels)) {
              updatedLabels[index] = {}
            }
            const label = track.getLabel(index)
            if (label) {
              updatedLabels[index][label.id] = label
            }
            itemIndices.add(index)
          }
          track.clearUpdatedIndices()
        }
      } else {
        if (!(drawable.item in updatedLabels)) {
          updatedLabels[drawable.item] = {}
        }
        updatedLabels[drawable.item][drawable.labelId] =
          drawable.labelState
        itemIndices.add(drawable.item)
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
          newTracks.push(track.state)
          itemIndices.add(drawable.item)
        }
      } else {
        if (!(drawable.item in updatedLabels)) {
          updatedLabels[drawable.item] = {}
        }
        updatedLabels[drawable.item][drawable.labelId] = drawable.labelState
        itemIndices.add(drawable.item)
      }
    }
  })

  updatedShapeDrawables.forEach((shape) => {
    if (!(shape.item in updatedShapes)) {
      updatedShapes[shape.item] = {}
    }
    updatedShapes[shape.item][shape.id] = shape.toState()
    itemIndices.add(shape.item)
  })

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
