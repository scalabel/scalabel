import { Link } from '@material-ui/core'
import { withStyles } from '@material-ui/core/styles'
import Table from '@material-ui/core/Table'
import TableBody from '@material-ui/core/TableBody'
import TableCell from '@material-ui/core/TableCell'
import TableHead from '@material-ui/core/TableHead'
import TableRow from '@material-ui/core/TableRow'
import Typography from '@material-ui/core/Typography'
import React from 'react'
import { toProject } from '../common/service'
import { User } from '../components/dashboard_worker'
import { tableStyles } from '../styles/dashboard'

interface DataTableClassType {
  /** root class */
  root: string
  /** row styles class */
  row: string
  /** header cell style */
  headerCell: string
}

interface HeaderSpecs {
  /** actual header string */
  header: string
  /** header alignment */
  align: 'left' | 'right' | 'center'
}

interface DataTableProps<DataType> {
  /** styles of Project Table */
  classes: DataTableClassType
  /** project header specifications */
  headers: HeaderSpecs[]
  /** data array to display */
  dataList: DataType[]
}

/**
 * class for generic data table, used for both worker and admin dashboard
 */
class DataTable<DataType extends string | User>
  extends React.Component<DataTableProps<DataType>> {
  public constructor (props: DataTableProps<DataType>) {
    super(props)
  }

  /**
   * renders the table
   */
  public render () {
    const { classes, dataList, headers } = this.props
    return (
      <Table size='medium'>
        <TableHead>
          <TableRow>
            {headers.map((value: HeaderSpecs) =>
              (<TableCell align={value.align} key={value.header}
                className={classes.headerCell}>
                {value.header}</TableCell>))}
          </TableRow>
        </TableHead>
        <TableBody>
          {dataList.map((value: DataType, index: number) => {
            let row: React.ReactFragment
            if (typeof value !== 'string') {
              row = (<React.Fragment>
                <TableCell align={'left'}>
                  <Typography variant={'body2'}>
                    {(value as User).email}
                  </Typography>
                </TableCell>
                <TableCell align={'right'}>
                  <Typography variant={'body2'}>
                    {(value as User).group}
                  </Typography>
                </TableCell>
              </React.Fragment>
              )
            } else {
              row = (<TableCell align={'left'}>
                <Typography>
                  <Link
                    variant={'body2'}
                    color={'inherit'}
                    onClick={() => {
                      toProject(value)
                    }}
                  >
                    {value}
                  </Link>
                </Typography>
              </TableCell>)
            }
            return (
              <TableRow key={index} className={
                index % 2 === 0 ? classes.row : ''
              }>
                {row}
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    )
  }
}

/** export DataTable */
export default withStyles(tableStyles)(DataTable)
