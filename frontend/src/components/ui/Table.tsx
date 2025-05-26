import React from 'react';
import { MoreHorizontalIcon } from 'lucide-react';

interface TableProps {
  children: React.ReactNode;
  className?: string;
}

interface TableHeaderProps {
  children: React.ReactNode;
  className?: string;
}

interface TableBodyProps {
  children: React.ReactNode;
  className?: string;
}

interface TableRowProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

interface TableHeadProps {
  children: React.ReactNode;
  className?: string;
  sortable?: boolean;
  sortDirection?: 'asc' | 'desc' | null;
  onSort?: () => void;
}

interface TableCellProps {
  children: React.ReactNode;
  className?: string;
}

export const Table: React.FC<TableProps> = ({ children, className = '' }) => {
  return (
    <div className="overflow-hidden">
      <table className={`min-w-full divide-y divide-gray-200 dark:divide-gray-700 ${className}`}>
        {children}
      </table>
    </div>
  );
};

export const TableHeader: React.FC<TableHeaderProps> = ({ children, className = '' }) => {
  return (
    <thead className={`bg-gray-50 dark:bg-gray-800 ${className}`}>
      {children}
    </thead>
  );
};

export const TableBody: React.FC<TableBodyProps> = ({ children, className = '' }) => {
  return (
    <tbody className={`bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700 ${className}`}>
      {children}
    </tbody>
  );
};

export const TableRow: React.FC<TableRowProps> = ({ children, className = '', onClick }) => {
  return (
    <tr
      className={`
        ${onClick ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800' : ''}
        ${className}
      `}
      onClick={onClick}
    >
      {children}
    </tr>
  );
};

export const TableHead: React.FC<TableHeadProps> = ({
  children,
  className = '',
  sortable = false,
  sortDirection = null,
  onSort,
}) => {
  return (
    <th
      className={`
        px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider
        ${sortable ? 'cursor-pointer select-none hover:text-gray-700 dark:hover:text-gray-300' : ''}
        ${className}
      `}
      onClick={sortable ? onSort : undefined}
    >
      <div className="flex items-center space-x-1">
        <span>{children}</span>
        {sortable && (
          <div className="flex flex-col">
            <div
              className={`w-0 h-0 border-l-2 border-r-2 border-b-2 border-transparent ${
                sortDirection === 'asc'
                  ? 'border-b-gray-600 dark:border-b-gray-300'
                  : 'border-b-gray-300 dark:border-b-gray-600'
              }`}
              style={{ borderBottomWidth: '4px', marginBottom: '1px' }}
            />
            <div
              className={`w-0 h-0 border-l-2 border-r-2 border-t-2 border-transparent ${
                sortDirection === 'desc'
                  ? 'border-t-gray-600 dark:border-t-gray-300'
                  : 'border-t-gray-300 dark:border-t-gray-600'
              }`}
              style={{ borderTopWidth: '4px' }}
            />
          </div>
        )}
      </div>
    </th>
  );
};

export const TableCell: React.FC<TableCellProps> = ({ children, className = '' }) => {
  return (
    <td className={`px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 ${className}`}>
      {children}
    </td>
  );
};

// Action menu component for table rows
interface TableActionsProps {
  children: React.ReactNode;
  className?: string;
}

export const TableActions: React.FC<TableActionsProps> = ({ children, className = '' }) => {
  return (
    <td className={`px-6 py-4 whitespace-nowrap text-right text-sm font-medium ${className}`}>
      <div className="flex items-center justify-end space-x-2">
        {children}
      </div>
    </td>
  );
};
