import { Link } from '@material-ui/core'
import Paper from '@material-ui/core/Paper'
import { withStyles } from '@material-ui/core/styles'
import Table from '@material-ui/core/Table'
import TableBody from '@material-ui/core/TableBody'
import TableCell from '@material-ui/core/TableCell'
import TableHead from '@material-ui/core/TableHead'
import TableRow from '@material-ui/core/TableRow'
import React from 'react'
import { getProjects, toProject } from '../common/service'
import { tableCellStyles } from '../styles/dashboard'

/* Retrieve data from backend */
const projectsToExpress = getProjects()

interface ClassType {
  /** root of project table */
  root: string

  /** table */
  table: string

  /** row */
  row: string
}

const ProjectTable = (props: {
  /** styles of Project Table */
  classes: ClassType;
}) => {
  const { classes } = props
  return (
    <Paper className={classes.root}>
      <Table className={classes.table}>
        <TableHead>
          <TableRow>
            <DashboardTableCell>Projects</DashboardTableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {projectsToExpress.map((row: string, i: number) => (
            <TableRow className={classes.row} key={i}>
              <DashboardTableCell component='th' scope='row'>
                <Link
                  component='button'
                  variant='body2'
                  color='inherit'
                  onClick={() => {
                    toProject(row)
                  }}
                >
                  {row}
                </Link>
              </DashboardTableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  )
}

const DashboardTableCell = withStyles(tableCellStyles)(TableCell)

/** export ProjectTable */
export default (ProjectTable)
