class Jpknn():
    
    def __init__(self,k):
        self.k = k

        
    def dist_euclidiana(self,item_white, lista_colorida):
        '''Calcula a distância euclidiana entre os pontos
    
        Calcula a distância euclidiana entre o item a ser classificado e todos os itens 
        já classificados.

        Args:
            item_white: Lista com os dados do investidor a ser classificado. Por exemplo:
                [45926320819, '', (5800., 4000., 1200., 200.)]
                Obs.: dentro da tupla pode ter um número n de dimensões.
            lista_colorida: Lista com os  dados dos investidores já classificados. Por exemplo:
                [[49212614633, 'Agressivo', (5900., 3000., 5100., 1800.)],...]


        Returns:
            Retorna uma lista de tuplas ordenadas de maneira crescente.
            Onde em cada tupla tem a distancia(float) e o indice(inteiro) 
            do elemento que foi comparado por exemplo:

            [(menor_distancia,indice_menor_distancia),...,(maior_distancia,indice_maior distancia)]
        '''


        dist = []
        for i in range(len(lista_colorida)):
            d = 0
            for j in range(len(item_white[2])):
                d +=  (item_white[2][j] - lista_colorida[i][2][j])**2
            dist.append((d**(.5), i))
        return sorted(dist)


    def perfil_inv(self,lista_dist, lista_colorida):
        
        '''Classifica os itens.

        Diz qual perfil o investidor provalmente possuirá.

        Args:
            k: Variável do tipo int, representando o número de vizinhos desejado para o KNN.
            lista_dist: Uma lista de tuplas, ordenadas de maneira crescente. Com a distancia
                de um investidor não classificado para os classificados.Por exemplo:
                [(menor_distancia,indice_menor_distancia),...,(maior_distancia,indice_maior_distancia)]
            lista_colorida: Lista de dados já classificados. Por exemplo:
                [[49212614633, 'Agressivo', (5900., 3000., 5100., 1800.)],...]

        Returns:
            Uma tupla com a quantidade de votos recebidas e qual o perfil do investidor
            por exemplo:

            (5, 'Conservador')
        '''

        cons = 0
        mod = 0
        agre = 0
        for i in range(self.k):
            indice = lista_dist[i][1]
            if lista_colorida[indice][1] == 'Conservador':
                cons += 1
            elif lista_colorida[indice][1] == 'Moderado':
                mod += 1
            else:
                agre += 1
            res_vote = sorted([(cons,'Conservador'),(mod,'Moderado'),(agre,'Agressivo')])
        return res_vote[-1]

    def predicao_perfil(self,lista_white,lista_colorida):
        '''Prediz qual perfil do investidor.

        Utiliza do modelo supervisionado de machine learning KNN, Indicando o possível perfil dos 
        novos investidores baseado nos perfis dos investidores já conhecidos

        Args:
            k: Variável do tipo int, representando o número de vizinhos desejado para o KNN.
            lista_white: Lista de dados dos investidores não classificados. Por exemplo:
                [[45926320819, '', (5800., 4000., 1200., 200.)],...].
                Obs.: dentro da tupla pode ter um número n de dimensões.
            lista_colorida: Lista de dados dos investidores já classificado. Por exemplo:
                [[49212614633, 'Agressivo', (5900., 3000., 5100., 1800.)],...].
                Obs.: dentro da tupla pode ter um número n de dimensões.

        Returns:
            Dicionário com a chave sendo identificador do investidor e o valor como sendo
            o perfil provável daquele investidor. Por exemplo:

            {45926320819: 'Conservador',
            52559670741: 'Conservador',
            59016004832: 'Conservador'}
            '''



        distance = []
        for perfil in lista_white:
            distance.append(self.dist_euclidiana(perfil,lista_colorida))

        itens_class ={}
        for i in range(len(lista_white)):
            res = self.perfil_inv(distance[i],lista_colorida)
            itens_class[lista_white[i][0]] = res[-1]

        return itens_class