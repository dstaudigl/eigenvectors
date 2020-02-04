#!/usr/bin/env python3

'''Eigen vector approximation to solve wave equation

Authors: David Staudigl, Medhi Garouachi
Date: 03.02.2020

This script computes the eigen values and eigen vectors of a given wave
equation numerically as well as analytically and compares the results.
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

def main():

    # # # # # # # # # # # # #
    #      Begin Setup      #
    # # # # # # # # # # # # #

    # Number of intervals along [0, 1] to interpolate with,
    # must be at least 6 for plotting:

    N = 75

    # Precision of analytical plots on [0, 1]:

    E = 10 ** 6

    # Set PRINT to True to print results:

    PRINT = False

    # Set SVG to True to save plots in SVG format:

    SVG = True

    # Set SHOW to True to show plots:

    SHOW = True

    # # # # # # # # # # # # #
    #       End Setup       #
    # # # # # # # # # # # # #

    # Falsify setup validity:

    if N <= 5:
        sys.exit( '[{}]: Need at least N = 5 intervals in setup.'
            .format( sys.argv[ 0 ] ) )

    # Create equidistant numpy.linspace along [0, 1]:

    x = np.linspace( 0, 1, N + 1, endpoint = True )

    # Create tridiagonal matrix A to approximate
    # second order central differential quotient:

    A_m = np.diag( 2 * np.ones( N - 1 ) )
    A_l = np.roll(
        np.diag(
            np.concatenate( ( -1 * np.ones( N - 2 ), [0] ) ) ), 1, axis = 0 )
    A_u = np.roll(
        np.diag(
            np.concatenate( ( -1 * np.ones( N - 2 ), [0] ) ) ), 1, axis = 1 )

    # A =: - 1 / h² tridiag( 1, -2, 1 )

    A = N ** 2 * ( A_m + A_l + A_u )

    # Solve eigen problem to find eigen values λ and eigen vectors v:

    numerical_eigen_values, numerical_eigen_vectors = np.linalg.eig( A )

    n_e_vec_len = len( numerical_eigen_vectors[ 0, : ] )
    n_e_val_len = len( numerical_eigen_values )

    # Sort by λ in ascending order:

    a = numerical_eigen_values.argsort()[ : : 1 ]
    numerical_eigen_vectors = numerical_eigen_vectors[ :, a ]
    numerical_eigen_values = numerical_eigen_values[ a ]

    # Compare results to the analytic eigen values λ = κ² and eigen vectors
    # ( v( xⱼ ) )ⱼ = ( sin( κ xⱼ ) )ⱼ, where κ ∊ ℕπ:

    analytic_eigen_values = np.array(
        [ k ** 2 * np.pi ** 2 for k in range( 1, N ) ] )
    analytic_eigen_vectors = np.array( [ np.array(
        [
            np.sin( j * np.pi * x[k] ) for k in range( 1, N )
        ] ) for j in range( 1, N ) ] )
    
    # Normalise analytic eigen vectors for comparison in plots:
    
    analytic_eigen_vectors_normalised \
        = np.array( [
            analytic_eigen_vectors[ k ] \
            / np.linalg.norm( analytic_eigen_vectors[ k ] ) \
            for k in range( 0, N - 1 )
        ] )
    
    # Print results if PRINT set to True:

    if PRINT == True:
        
        print( 'Equidistant points along [0, 1]:' )
        
        print( x )

        print( 'Matrix A:' )

        print( A )

        print( 'Computed the first {} analytical eigen values:'
             .format( n_e_val_len ) )

        print( analytic_eigen_values )

        print( 'Computed {} numerical eigen values:'
             .format( n_e_val_len ) )

        print( numerical_eigen_values )

        print( 'Computed {} corresponding numerical eigen vectors:'
            .format( n_e_vec_len ) )

        print( numerical_eigen_vectors )
        
        print( 'Exemplary falsification for first numerical eigen vector:' )
        
        print( np.dot( A[ :, : ], numerical_eigen_vectors[ :, 0 ] ) )
        
        print( numerical_eigen_vectors[ :, 0 ] * numerical_eigen_values[ 0 ] )

    # Generate plots for individual results and comparison,
    # using precision E for analytic plots:

    x_plot = np.linspace( 0, 1, E + 1, endpoint = True )
    
    # For the fifth analytical eigen vector, which shall be used
    # for a plot comparison, we require a manual coefficient of -1:
    
    analytic_eigen_vectors_normalised[ 4 ] \
        = - analytic_eigen_vectors_normalised[ 4 ]

    # Plot the first \ fifth analytic eigen vector, corresponding to the
    # first \ fifth analytic eigen value π² \ 25 π², along [0, 1]:

    fig_analytic, ( ( ax_analytic_first, ax_analytic_fifth ) ) \
        = plt.subplots( 2, 1, figsize = ( 9, 9 ) )

    ax_analytic_first.plot(
        x_plot,np.sin( np.pi * x_plot ) \
            / np.linalg.norm( analytic_eigen_vectors[ 0 ] ), '-',
        color = 'royalblue', label = '$\kappa = \pi$' )
    ax_analytic_first.plot(
        x[ 1 : -1 ], analytic_eigen_vectors_normalised[ 0 ], '.',
        color = 'royalblue' )
    ax_analytic_fifth.plot(
        x_plot, - np.sin( 5 * np.pi * x_plot ) \
            / np.linalg.norm( analytic_eigen_vectors[ 4 ] ), '-',
        color = 'royalblue', label = '$\kappa = 5 \pi$' )
    ax_analytic_fifth.plot(
        x[ 1 : -1 ], analytic_eigen_vectors_normalised[ 4 ], '.',
        color = 'royalblue' )

    ax_analytic_first.legend()
    ax_analytic_first.set_xlabel( '$x [1]$' )
    ax_analytic_first.set_ylabel( '$v( x ) [1]$' )

    ax_analytic_fifth.legend()
    ax_analytic_fifth.set_xlabel( '$x [1]$' )
    ax_analytic_fifth.set_ylabel( '$v( x ) [1]$' )

    # Plot the first \ fifth numeric eigen vector along [0, 1]:

    fig_numeric, ( ( ax_numeric_first, ax_numeric_fifth ) ) \
        = plt.subplots( 2, 1, figsize = ( 9, 9 ) )

    ax_numeric_first.plot( x[ 1 : -1 ], numerical_eigen_vectors[ :, 0 ], '.',
        color = 'orangered', label = '$1. num$' )
    ax_numeric_fifth.plot( x[ 1 : -1 ], numerical_eigen_vectors[ :, 4 ], '.',
        color = 'orangered', label = '$5. num$' )

    ax_numeric_first.legend()
    ax_numeric_first.set_xlabel( '$x [1]$' )
    ax_numeric_first.set_ylabel( '$v( x ) [1]$' )

    ax_numeric_fifth.legend()
    ax_numeric_fifth.set_xlabel( '$x [1]$' )
    ax_numeric_fifth.set_ylabel( '$v( x ) [1]$' )

    # Plot a comparison:

    fig_comp, ( ( ax_comp_first, ax_comp_fifth ) ) \
        = plt.subplots( 2, 1, figsize = ( 9, 9 ) )

    ax_comp_first.plot( x_plot, np.sin( np.pi * x_plot ) \
            / np.linalg.norm( analytic_eigen_vectors[ 0 ] ), '-',
        color = 'royalblue', label = '$\kappa = \pi$' )
    ax_comp_first.plot(
        x[ 1 : -1 ], analytic_eigen_vectors_normalised[ 0 ], '.',
        color = 'royalblue' )
    ax_comp_fifth.plot( x_plot, - np.sin( 5 * np.pi * x_plot ) \
            / np.linalg.norm( analytic_eigen_vectors[ 4 ] ), '-',
        color = 'royalblue', label = '$\kappa = 5 \pi$' )
    ax_comp_fifth.plot(
        x[ 1 : -1 ], analytic_eigen_vectors_normalised[ 4 ], '.',
        color = 'royalblue' )
    ax_comp_first.plot( x[ 1 : -1 ], numerical_eigen_vectors[ :, 0 ], '.',
        color = 'orangered', label = '$1. num$' )
    ax_comp_fifth.plot( x[ 1 : -1 ], numerical_eigen_vectors[ :, 4 ], '.',
        color = 'orangered', label = '$5. num$' )

    ax_comp_first.legend()
    ax_comp_first.set_xlabel( '$x [1]$' )
    ax_comp_first.set_ylabel( '$v( x ) [1]$' )

    ax_comp_fifth.legend()
    ax_comp_fifth.set_xlabel( '$x [1]$' )
    ax_comp_fifth.set_ylabel( '$v( x ) [1]$' )


    # Handle subplot margins:

    fig_analytic.tight_layout()

    # Display and / or save figure(s):

    if SVG == True:
        fig_analytic.savefig( 'plots/analytic.svg', format = 'svg' )
        fig_numeric.savefig( 'plots/numeric.svg', format = 'svg' )
        fig_comp.savefig( 'plots/comparison.svg', format = 'svg' )

    if SHOW == True:
        plt.show()

if __name__ == '__main__':
    main()
