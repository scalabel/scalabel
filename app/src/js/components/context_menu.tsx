import { Grid } from '@material-ui/core'
import Divider from '@material-ui/core/Divider'
import TextField from '@material-ui/core/TextField'
import { Autocomplete } from '@material-ui/lab'
import * as React from 'react'
import { changeSelect } from '../action/common'
import { changeSelectedLabelsCategories } from '../action/select'
import Session from '../common/session'
import { AttributeSelector } from './attribute_selector'

/** Context menu class */
export class ContextMenu extends React.Component {
  constructor () {
    super({})
  }

  /** Render context menu */
  public render () {
    const categories = Session.getState().task.config.categories
    const selectedLabel = Session.label3dList.selectedLabel
    return (
      <Grid
        justify={'flex-start'}
        container
        direction='row'
      >
        <Autocomplete
          options={categories}
          getOptionLabel={(x) => x}
          id='contextMenuCategories'
          style={{ width: '100%' }}
          renderInput={
            (params) =>
              <TextField
                {...params}
                margin='normal'
                label='Category'
                fullWidth
              />
          }
          disabled={!selectedLabel}
          onChange={(_e: React.ChangeEvent<{}>, val: string | null) => {
            if (val) {
              const state = Session.getState()
              const categoryId =
                state.task.config.categories.indexOf(val)
              Session.dispatch(changeSelect({ category: categoryId }))
              // update categories if any labels are selected
              if (Object.keys(state.user.select.labels).length > 0) {
                Session.dispatch(changeSelectedLabelsCategories(
                  state, [categoryId]
                ))
              }
            }
          }}
          value={(selectedLabel) ? categories[selectedLabel.category[0]] : null}
        />
        <Divider variant='middle' />
        <AttributeSelector/>
      </Grid>
    )
  }
}
