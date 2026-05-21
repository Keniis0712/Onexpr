"""Entry point: `python -m store` (or, after bundling, `python app.py`).

Exercises:
  - `from store import api`
  - class instantiation + method chains
  - closure passed across modules (search lambda)
  - module global mutation visible from another module's globals()
"""
from store import api


def main():
    db = api.Database()
    db.put('alice', {'role': 'admin', 'age': 32})
    db.put('bob', {'role': 'user', 'age': 25})
    db.put('cara', {'role': 'user', 'age': 41})

    print('count:', db.count())
    print('repr:', repr(db))
    print('alice:', db.get('alice'))

    admins = db.search(lambda v: v['role'] == 'admin')
    print('admins:', sorted(admins))

    older = db.search(lambda v: v['age'] >= 30)
    print('age>=30:', sorted(older))

    db.delete('bob')
    db.count.cache_clear()
    print('after delete count:', db.count())
    print('latest id:', api.latest_id())


if __name__ == '__main__':
    main()
