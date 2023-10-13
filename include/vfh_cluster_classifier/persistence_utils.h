#ifndef PERSISTENCE_UTILS_H
#define PERSISTENCE_UTILS_H

#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

namespace PersistenceUtils
{
    inline bool
    writeMatrixToFile (std::string file, Eigen::Matrix4f & matrix)
    {
      std::ofstream out (file.c_str ());
      if (!out)
      {
        std::cout << "Cannot open file.\n";
        return false;
      }

      for (size_t i = 0; i < 4; i++)
      {
        for (size_t j = 0; j < 4; j++)
        {
          out << matrix (i, j);
          if (!(i == 3 && j == 3))
            out << " ";
        }
      }
      out.close ();

      return true;
    }

    inline bool
    readMatrixFromFile (std::string file, Eigen::Matrix4f & matrix)
    {

      std::ifstream in;
      in.open (file.c_str (), std::ifstream::in);

      char linebuf[1024];
      in.getline (linebuf, 1024);
      std::string line (linebuf);
      std::vector < std::string > strs_2;
      boost::split (strs_2, line, boost::is_any_of (" "));

      for (int i = 0; i < 16; i++)
      {
        matrix (i % 4, i / 4) = static_cast<float> (atof (strs_2[i].c_str ()));
      }

      return true;
    }

    inline bool
    writeCentroidToFile (std::string file, Eigen::Vector3f & centroid)
    {
      std::ofstream out (file.c_str ());
      if (!out)
      {
        std::cout << "Cannot open file.\n";
        return false;
      }

      out << centroid[0] << " " << centroid[1] << " " << centroid[2] << std::endl;
      out.close ();

      return true;
    }

    inline bool getCentroidFromFile(std::string file, Eigen::Vector3f & centroid)
    {
        std::ifstream in;
        in.open (file.c_str (), std::ifstream::in);

        char linebuf[256];
        in.getline (linebuf, 256);
        std::string line (linebuf);
        std::vector < std::string > strs;
        boost::split (strs, line, boost::is_any_of (" "));
        centroid[0] = static_cast<float> (atof (strs[0].c_str ()));
        centroid[1] = static_cast<float> (atof (strs[1].c_str ()));
        centroid[2] = static_cast<float> (atof (strs[2].c_str ()));

        return true;
    }

    inline bool readGroundTruthFromFile (std::string file, std::string model, Eigen::Matrix4f & matrix)
    {
        std::ifstream in ( file.c_str());

        if(in.is_open())
        {
            while(in.good())
            {
                string line;
                getline(in, line);

                if(line.substr(0, model.size()) == model)
                {
                    int row_ind = 0;

                    while(row_ind < 4)
                    {
                        getline(in, line);

                        std::vector<string> strs;
                        boost::split(strs, line, boost::is_any_of(" "));

                        for(int j = 0; j < 4; j++)
                        {
                            matrix(row_ind, j) = static_cast<float> (atof (strs[j].c_str ()));
                        }

                        row_ind++;
                    }

                    in.close();
                    return true;
                }

            }

            in.close();
            return false;
        }
    }

    inline bool modelPresents (std::string file, std::string model)
    {
        std::ifstream in(file.c_str());

        if(in.is_open())
        {
            while(in.good())
            {
                string line;
                getline(in, line);

                if(line.substr(0, model.size()) == model)
                {
                    in.close();
                    return true;
                }
            }
            in.close();
            return false;
        }
    }
}

#endif // PERSISTENCE_UTILS_H