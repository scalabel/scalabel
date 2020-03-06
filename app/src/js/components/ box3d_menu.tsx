import { FormLabel, Grid } from '@material-ui/core'
import TextField from '@material-ui/core/TextField'
import * as React from 'react'
import * as THREE from 'three'
import Session from '../common/session'
import Label3D from '../drawable/3d/label3d'

interface Props {
  /** Label to edit */
  label: Label3D
}

/** Create form for editing vector 3 values */
function makeVector3Form (
  vector: THREE.Vector3,
  callback: (v: THREE.Vector3) => void = () => { return }
): JSX.Element {
  return (
    <Grid
      justify={'flex-start'}
      container
      direction='row'
    >
      <TextField
        label={'x'}
        style={{ width: '33%' }}
        defaultValue={vector.x.toFixed(2)}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
          const newX = Number(e.target.value)
          if (!isNaN(newX)) {
            vector.x = newX
            callback(vector)
          }
        }}
      />
      <TextField
        label={'y'}
        style={{ width: '33%' }}
        defaultValue={vector.y.toFixed(2)}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
          const newY = Number(e.target.value)
          if (!isNaN(newY)) {
            vector.y = newY
            callback(vector)
          }
        }}
      />
      <TextField
        label={'z'}
        style={{ width: '33%' }}
        defaultValue={vector.z.toFixed(2)}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
          const newZ = Number(e.target.value)
          if (!isNaN(newZ)) {
            vector.z = newZ
            callback(vector)
          }
        }}
      />
    </Grid>
  )
}

/** Menu for editing box 3d label */
export class Box3dMenu extends React.Component<Props> {
  constructor (props: Props) {
    super(props)
  }

  /** Render context menu */
  public render () {
    const label = this.props.label
    const center = label.center
    const orientation =
      (new THREE.Euler()).setFromQuaternion(label.orientation).toVector3()
    const size = label.size
    return (
      <div>
        <div style={{ marginBottom: '5px' }}>
          <FormLabel>Center</FormLabel>
          {makeVector3Form(center, (v) => {
            label.setCenter(v)
            Session.label3dList.onDrawableUpdate()
          })}
        </div>
        <div style={{ marginBottom: '5px' }}>
          <FormLabel>Size</FormLabel>
          {makeVector3Form(size, (v) => {
            label.setSize(v)
            Session.label3dList.onDrawableUpdate()
          })}
        </div>
        <div style={{ marginBottom: '5px' }}>
          <FormLabel>Orientation</FormLabel>
          {makeVector3Form(orientation, (v) => {
            label.setOrientation(
              (new THREE.Quaternion()).setFromEuler(
                (new THREE.Euler()).setFromVector3(v)
              )
            )
            Session.label3dList.onDrawableUpdate()
          })}
        </div>
      </div>
    )
  }
}
