#ifndef MAT2D_H_
#define MAT2D_H_

#include <vector>
#include <iostream>

template <class T>
class mat2D
{
public:
	int rows;
	int cols;

	mat2D(int rows, int cols)
	{
		this->rows = rows;
		this->cols = cols;
		this->vect = std::vector<T>(rows*cols, 0);
	}

	~mat2D()
	{
	}

	/*
	T Read(int pos)
	{
		return this->vect[pos];
	}
	*/

	inline T Read(int row, int col)
	{
		return this->vect[row*cols+col];
	}
	/*
	void Write(int pos, T val)
	{
		this->vect[pos] = val;
	}
	*/

	inline void Write(int row, int col, T val)
	{
		this->vect[row*cols+col] = val;
	}

	void Print(int rowFrom, int rowTo, int colFrom, int colTo)
	{
		std::cout << "\n";
		for (int r=rowFrom; r<=rowTo; r++)
		{	
			for (int c=colFrom; c<=colTo; c++)
			{
				std::cout << this->Read(r, c) << " ";
			}
			std::cout << "\n";
		}
	}

	void Print()
	{
		Print(0, this->rows-1, 0, this->cols-1);
	}

	mat2D<T> * FlipLR()
	{
		mat2D<T> * flipped = new mat2D<T>(this->rows, this->cols);
		for (int r=0; r < this->rows; r++)
		{
			for (int c=0; c < this->cols; c++)
			{
				flipped->Write(r, this->cols-1-c, this->Read(r,c));
			}
		}
		return flipped;
	}

	mat2D<T> * FlipUD()
	{
		mat2D<T> * flipped = new mat2D<T>(this->rows, this->cols);
		for (int r=0; r < this->rows; r++)
		{
			for (int c=0; c < this->cols; c++)
			{
				flipped->Write(this->rows-1-r, c, this->Read(r,c));
			}
		}
		return flipped;
	}

	mat2D<T> * Transpose()
	{
		mat2D<T> * flipped = new mat2D<T>(this->cols, this->rows);
		for (int r=0; r < this->rows; r++)
		{
			for (int c=0; c < this->cols; c++)
			{
				flipped->Write(c, r, this->Read(r,c));
			}
		}
		return flipped;
	}

	mat2D<T>* Correlation(mat2D<int>* kernel)
	{
		mat2D<T> *residual = new mat2D<T>(this->rows-kernel->rows+1, this->cols-kernel->cols+1);
		for (int ir=0; ir < (this->rows - kernel->rows + 1); ir++)
		{
			for (int ic=0; ic < (this->cols - kernel->cols + 1); ic++)
			{
				T convVal = 0;
				for (int kr=0; kr < kernel->rows; kr++)
				{
					for (int kc=0; kc < kernel->cols; kc++)
					{
						convVal = convVal + this->Read(ir+kr, ic+kc) * ((T)kernel->Read(kr, kc));
					}
				}
				residual->Write(ir, ic, convVal);
			}
		}
		return residual;
	}

	mat2D<float>* Correlation_2float(mat2D<float>* kernel)
	{
		mat2D<float> *residual = new mat2D<float>(this->rows-kernel->rows+1, this->cols-kernel->cols+1);
		for (int ir=0; ir < (this->rows - kernel->rows + 1); ir++)
		{
			for (int ic=0; ic < (this->cols - kernel->cols + 1); ic++)
			{
				float convVal = 0;
				for (int kr=0; kr < kernel->rows; kr++)
				{
					for (int kc=0; kc < kernel->cols; kc++)
					{
						convVal = convVal + this->Read(ir+kr, ic+kc) * kernel->Read(kr, kc);
					}
				}
				residual->Write(ir, ic, convVal);
			}
		}
		return residual;
	}

	std::vector<T> vect;
};

#endif
